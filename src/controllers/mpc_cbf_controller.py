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

    def solve(self, u0_warm, current_state, reference_trajectory, obstacle_pos=None):
        u_warm = u0_warm.reshape((self.N, 2))
        x_bar, A_seq, B_seq = self._predict_trajectory(current_state, u_warm)

        # 1. Matriz de proyección S_u
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

        # 2. Matrices Q y R (sin penalización en theta)
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

        # 3. Vector de error E
        E = np.zeros(4 * self.N)
        for i in range(self.N):
            ref_x, ref_y, ref_v = reference_trajectory[i]
            E[4*i] = x_bar[i + 1, 0] - ref_x
            E[4*i + 1] = x_bar[i + 1, 1] - ref_y
            E[4*i + 2] = x_bar[i + 1, 2] - ref_v

        # 4. Formulación de Hessiana P y gradiente q
        P_dense = S_u.T @ Q @ S_u + R.toarray() + D_diff.T @ R_d @ D_diff
        P = sparse.csc_matrix(P_dense)
        q = S_u.T @ Q @ E - D_diff.T @ R_d @ D_diff @ u_warm.flatten()

        # 5. Límites físicos de control (-1.0 <= u <= 1.0)
        A_bounds = sparse.eye(2 * self.N, format='csc')
        l_bounds = -np.ones(2 * self.N)
        u_bounds = np.ones(2 * self.N)

        A_list = [A_bounds]
        l_list = [l_bounds]
        u_list = [u_bounds]

        # 6. Restricción CBF
        if obstacle_pos is not None:
            x_obs, y_obs = obstacle_pos[0], obstacle_pos[1]
            A_cbf = np.zeros((self.N, 2 * self.N))
            l_cbf = np.zeros(self.N)
            u_cbf = np.full(self.N, np.inf)

            x_curr = current_state.copy()
            for k in range(self.N):
                v_k = max(x_curr[2], 0.1)
                R_margin = 2.5 + 0.3 * v_k

                dh_dx = 2 * (x_curr[0] - x_obs)
                dh_dy = 2 * (x_curr[1] - y_obs)
                grad_h = np.array([dh_dx, dh_dy, 0.0, 0.0])

                A_cbf[k, :] = grad_h @ S_u[4*k:4*(k+1), :]
                h_curr = (x_curr[0] - x_obs)**2 + (x_curr[1] - y_obs)**2 - R_margin**2
                l_cbf[k] = -GAMMA_CBF * h_curr

                x_curr = KinematicBicycleModel.step(x_curr, u_warm[k])

            A_list.append(sparse.csc_matrix(A_cbf))
            l_list.append(l_cbf)
            u_list.append(u_cbf)

        A_qp = sparse.vstack(A_list, format='csc')
        l_qp = np.hstack(l_list)
        u_qp = np.hstack(u_list)

        # 7. Resolver en OSQP
        prob = osqp.OSQP()
        prob.setup(P, q, A_qp, l_qp, u_qp, verbose=False, eps_abs=1e-3, eps_rel=1e-3)
        res = prob.solve()

        if res.info.status == 'solved':
            u_opt = res.x.reshape((self.N, 2))
            return u_opt, res.x

        u_brake = u_warm.copy()
        u_brake[:, 1] = -1.0
        return u_brake, u_brake.flatten()