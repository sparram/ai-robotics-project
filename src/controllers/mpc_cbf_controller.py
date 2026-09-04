import numpy as np
import osqp
from scipy import sparse
from config import N, DT, GAMMA_CBF
from models.kinematic import KinematicBicycleModel

class MPC_CBF:
    def __init__(self, horizon=N):
        self.N = horizon

        # Define quadratic weights
        self.w_pos = 0.60 / (2.0 ** 2)
        self.w_spd = 0.20 / (5.0 ** 2)
        self.w_ctrl_steer = 0.20 / (0.5 ** 2)
        self.w_ctrl_acc = 0.20 / (1.0 ** 2)
        self.w_dctrl_steer = 0.20 / (0.1 ** 2)
        self.w_dctrl_acc = 0.20 / (2.0 ** 2)

    # Extract position and velocities of the nearby vehicles in a 18m radius
    def _extract_obstacles(self, env, vehicle_pos, max_dist=18.0):
        obstacles_list = []
        vehicles = env.engine.traffic_manager.vehicles
        for v in vehicles:
            if v != env.agent:
                dist = np.linalg.norm(v.position - vehicle_pos)
                if dist < max_dist:
                    v_m_s = max(v.speed_km_h / 3.6, 0.0)
                    heading = v.heading_theta
                    vx = v_m_s * np.cos(heading)
                    vy = v_m_s * np.sin(heading)
                    obstacles_list.append(np.array([v.position[0], v.position[1], vx, vy]))
        return obstacles_list

    # Get index of a safe lane, returns None if there's no safe lane
    def _get_lane_safe(self, road_net, road_index):
        if road_index[2] < 0:
            return None
        try:
            return road_net.get_lane(road_index)
        except (KeyError, AttributeError, IndexError):
            return None

    # Determines if the chosen lane is blocked according to the actual state and nearby vehicles position
    # Returns True if it's blocked, returns False if not
    def _is_lane_blocked(self, lane_obj, state_real, obstacles_list):
        if lane_obj is None:
            return True
        s_e, _ = lane_obj.local_coordinates(state_real[:2])
        for obs in obstacles_list:
            s_o, lat_o = lane_obj.local_coordinates(obs[:2])
            d_fwd = s_o - s_e
            if 0.0 < d_fwd < 18.0 and abs(lat_o) < 1.5:
                return True
        return False

    # Generates a reference trajectory for the MPC using Frenet coordinate system
    def _generate_ref_trajectory(self, target_lane, state_real, should_stop):
        ref_trajectory = []
        s_ego_target, _ = target_lane.local_coordinates(state_real[:2])
        is_curved = abs(target_lane.heading_theta_at(s_ego_target + 5) - target_lane.heading_theta_at(s_ego_target)) > 0.1
        base_speed = 5.0 if is_curved else 8.0
        target_speed = 0.0 if should_stop else base_speed

        for i in range(self.N):
            target_s = s_ego_target + target_speed * (i + 1) * DT
            ref_x, ref_y = target_lane.position(target_s, 0.0)
            ref_trajectory.append((ref_x, ref_y, target_speed))

        return ref_trajectory

    # METADRIVE CONTROLLER: Returns the optimal action
    def get_action(self, env, state_real, u0_warm):
        # Extract ego vehicle and nearby obstacles
        vehicle = env.agent
        obstacles_list = self._extract_obstacles(env, vehicle.position)

        # Choose a safe lane. If there are no safe lanes, activates a stop trigger
        road_net = env.engine.map_manager.current_map.road_network
        current_road = vehicle.lane_index
        current_lane = vehicle.lane

        target_lane = current_lane
        should_stop = False

        if self._is_lane_blocked(current_lane, state_real, obstacles_list):
            left_index = (current_road[0], current_road[1], current_road[2] - 1)
            right_index = (current_road[0], current_road[1], current_road[2] + 1)

            left_lane = self._get_lane_safe(road_net, left_index)
            right_lane = self._get_lane_safe(road_net, right_index)

            if left_lane is not None and not self._is_lane_blocked(left_lane, state_real, obstacles_list):
                target_lane = left_lane
            elif right_lane is not None and not self._is_lane_blocked(right_lane, state_real, obstacles_list):
                target_lane = right_lane
            else:
                should_stop = True

        # Generate the reference trajectory along the chosen lane
        ref_trajectory = self._generate_ref_trajectory(target_lane, state_real, should_stop)

        # MPC CORE: Solve the MPC optimization problem
        u_nom_seq, u0_warm_flat = self.solve(
            u0_warm, state_real, ref_trajectory, obstacles_list=obstacles_list
        )

        # Update the sequence of actions
        u_nom_first = u_nom_seq[0]
        u0_warm_next = np.roll(u0_warm_flat, -2)
        u0_warm_next[-2:] = u0_warm_next[-4:-2]

        # Returns the optimal action, the updated base control sequence and the obstacle list
        return u_nom_first, u0_warm_next, obstacles_list

    # Predicts the evolution of the system given the actual sequence of actions
    # Returns the predicted trajectory and the Jacobians to linealize the dynamics on it
    def _predict_trajectory(self, current_state, u_seq):
        x_bar = np.zeros((self.N + 1, 4))
        x_bar[0] = current_state
        A_seq, B_seq = [], []

        for i in range(self.N):
            A = KinematicBicycleModel.jacobian_x(x_bar[i], u_seq[i])
            B = KinematicBicycleModel.jacobian_u(x_bar[i], u_seq[i])
            A_seq.append(A)
            B_seq.append(B)
            x_bar[i + 1] = KinematicBicycleModel.step(x_bar[i], u_seq[i])

        return x_bar, A_seq, B_seq

    # CORE MPC SOLVER: Solve the MPC Optimization problem via LTV approach
    # The optimization problem becomes a QP problem, solvable with OSQP
    def solve(self, u0_warm, current_state, reference_trajectory, obstacles_list=None):
        u_warm = u0_warm.reshape((self.N, 2))
        u_warm_flat = u_warm.flatten()
        x_bar, A_seq, B_seq = self._predict_trajectory(current_state, u_warm)

        # 1. Matriz de proyección S_u (4N x 2N)
        S_u = np.zeros((4 * self.N, 2 * self.N))
        for k in range(self.N):
            for j in range(k + 1):
                if k == j:
                    S_u[4*k:4*(k+1), 2*j:2*(j+1)] = B_seq[j]
                else:
                    A_prod = np.eye(4)
                    for m in range(j + 1, k + 1):
                        A_prod = A_seq[m] @ A_prod
                    S_u[4*k:4*(k+1), 2*j:2*(j+1)] = A_prod @ B_seq[j]

        # 2. Matrices de costo Q y R
        Q_block = np.diag([self.w_pos, self.w_pos, self.w_spd, 0.0])
        Q = sparse.kron(sparse.eye(self.N), Q_block)

        R_block = np.diag([self.w_ctrl_steer, self.w_ctrl_acc])
        R = sparse.kron(sparse.eye(self.N), R_block)

        D_diff = np.zeros((2 * (self.N - 1), 2 * self.N))
        R_d_diag = np.tile([self.w_dctrl_steer, self.w_dctrl_acc], self.N - 1)
        R_d = sparse.diags(R_d_diag)

        for i in range(self.N - 1):
            D_diff[2*i:2*(i+1), 2*i:2*(i+1)] = -np.eye(2)
            D_diff[2*i:2*(i+1), 2*(i+1):2*(i+2)] = np.eye(2)

        # 3. Vector de error E compensado por linealización en u_warm
        E_base = np.zeros(4 * self.N)
        for i in range(self.N):
            ref_x, ref_y, ref_v = reference_trajectory[i]
            E_base[4*i]     = x_bar[i + 1, 0] - ref_x
            E_base[4*i + 1] = x_bar[i + 1, 1] - ref_y
            E_base[4*i + 2] = x_bar[i + 1, 2] - ref_v

        E = E_base - S_u @ u_warm_flat

        # 4. Hessiana P y gradiente q
        P_dense = S_u.T @ Q @ S_u + R.toarray() + D_diff.T @ R_d @ D_diff
        P = sparse.csc_matrix(P_dense)
        q = S_u.T @ Q @ E - D_diff.T @ R_d @ D_diff @ u_warm_flat

        # 5. Límites físicos (-1.0 <= u <= 1.0)
        A_bounds = sparse.eye(2 * self.N, format='csc')
        l_bounds = -np.ones(2 * self.N)
        u_bounds = np.ones(2 * self.N)

        A_list = [A_bounds]
        l_list = [l_bounds]
        u_list = [u_bounds]

        # 6. RESTRICCIONES MULTI-CBF CON PROPAGACIÓN DINÁMICA
        if obstacles_list is not None and len(obstacles_list) > 0:
            M = len(obstacles_list)
            A_cbf = np.zeros((M * self.N, 2 * self.N))
            l_cbf = np.zeros(M * self.N)
            u_cbf = np.full(M * self.N, np.inf)

            row_idx = 0
            for obs in obstacles_list:
                x_obs0, y_obs0 = obs[0], obs[1]
                vx_obs = obs[2] if len(obs) > 2 else 0.0
                vy_obs = obs[3] if len(obs) > 3 else 0.0

                for k in range(self.N):
                    obs_xk = x_obs0 + vx_obs * (k * DT)
                    obs_yk = y_obs0 + vy_obs * (k * DT)
                    obs_xk1 = x_obs0 + vx_obs * ((k + 1) * DT)
                    obs_yk1 = y_obs0 + vy_obs * ((k + 1) * DT)

                    x_k = x_bar[k]
                    x_k1 = x_bar[k + 1]

                    v_k1 = max(x_k1[2], 0.1)
                    R_margin = 1.4 + 0.2 * v_k1

                    h_k = (x_k[0] - obs_xk)**2 + (x_k[1] - obs_yk)**2 - R_margin**2
                    h_k1 = (x_k1[0] - obs_xk1)**2 + (x_k1[1] - obs_yk1)**2 - R_margin**2

                    dh_dx = 2 * (x_k1[0] - obs_xk1)
                    dh_dy = 2 * (x_k1[1] - obs_yk1)
                    grad_h_k1 = np.array([dh_dx, dh_dy, 0.0, 0.0])

                    S_u_k1 = S_u[4*k:4*(k+1), :]
                    A_cbf[row_idx, :] = grad_h_k1 @ S_u_k1

                    l_cbf[row_idx] = (1.0 - GAMMA_CBF) * h_k - h_k1 + A_cbf[row_idx, :] @ u_warm_flat
                    row_idx += 1

            A_list.append(sparse.csc_matrix(A_cbf))
            l_list.append(l_cbf)
            u_list.append(u_cbf)

        A_qp = sparse.vstack(A_list, format='csc')
        l_qp = np.hstack(l_list)
        u_qp = np.hstack(u_list)

        # 7. Resolver con OSQP
        prob = osqp.OSQP()
        prob.setup(P, q, A_qp, l_qp, u_qp, verbose=False, eps_abs=1e-3, eps_rel=1e-3)
        res = prob.solve()

        if res.info.status == 'solved':
            u_opt = res.x.reshape((self.N, 2))
            return u_opt, res.x

        # 8. Estrategia de respaldo (Fallback)
        u_fallback = u_warm.copy()
        v_actual = current_state[2]

        if v_actual < 0.8:
            u_fallback[:, 0] = 0.4    # Giro suave para salir del bloqueo
            u_fallback[:, 1] = 0.15   # Impulso suave
        else:
            u_fallback[:, 1] = -1.0   # Frenado de emergencia

        return u_fallback, u_fallback.flatten()