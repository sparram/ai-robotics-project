import numpy as np
import osqp
from scipy import sparse
from config import N, DT, GAMMA_CBF
from models.kinematic import KinematicBicycleModel

class MPC_CBF:
    def __init__(self, horizon=N):
        self.N = horizon

        # Pesos cuadráticos normalizados
        self.w_pos = 0.60 / (2.0 ** 2)
        self.w_spd = 0.20 / (5.0 ** 2)
        self.w_ctrl_steer = 0.20 / (0.5 ** 2)
        self.w_ctrl_acc = 0.20 / (1.0 ** 2)
        self.w_dctrl_steer = 0.20 / (0.1 ** 2)
        self.w_dctrl_acc = 0.20 / (2.0 ** 2)

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

        # Restamos S_u @ u_warm_flat para corregir la expansión de Taylor del estado
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

        # 6. RESTRICCIONES MULTI-CBF EN TIEMPO DISCRETO
        if obstacles_list is not None and len(obstacles_list) > 0:
            M = len(obstacles_list)
            A_cbf = np.zeros((M * self.N, 2 * self.N))
            l_cbf = np.zeros(M * self.N)
            u_cbf = np.full(M * self.N, np.inf)

            row_idx = 0
            for obs_pos in obstacles_list:
                x_obs, y_obs = obs_pos[0], obs_pos[1]

                for k in range(self.N):
                    # Estados predichos en paso k y paso k+1
                    x_k = x_bar[k]
                    x_k1 = x_bar[k + 1]

                    v_k1 = max(x_k1[2], 0.1)
                    R_margin = 1.4 + 0.2 * v_k1

                    # h(x) evaluado en k y k+1
                    h_k = (x_k[0] - x_obs)**2 + (x_k[1] - y_obs)**2 - R_margin**2
                    h_k1 = (x_k1[0] - x_obs)**2 + (x_k1[1] - y_obs)**2 - R_margin**2

                    # Gradiente evaluado en el estado k+1
                    dh_dx = 2 * (x_k1[0] - x_obs)
                    dh_dy = 2 * (x_k1[1] - y_obs)
                    grad_h_k1 = np.array([dh_dx, dh_dy, 0.0, 0.0])

                    S_u_k1 = S_u[4*k:4*(k+1), :]
                    A_cbf[row_idx, :] = grad_h_k1 @ S_u_k1

                    # Condición discreta: h(x_{k+1}) >= (1 - gamma) * h(x_k)
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

        # 8. ESTRATEGIA DE MANEJO DE INFACTIBILIDAD (FALLBACK)
        u_fallback = u_warm.copy()
        v_actual = current_state[2]

        if v_actual < 0.8:
            u_fallback[:, 0] = 0.4   # Giro suave para salir del bloqueo
            u_fallback[:, 1] = 0.15  # Impulso suave
        else:
            u_fallback[:, 1] = -1.0  # Frenado de emergencia

        return u_fallback, u_fallback.flatten()