import numpy as np
from scipy.optimize import minimize
from config import N, DT, GAMMA_CBF
from models.kinematic import KinematicBicycleModel

class MPC_CBF:
    def __init__(self, horizon=N):
        self.N = horizon

    def _predict_trajectory(self, current_state, u_seq):
        """
        Predice la trayectoria nominal (x_bar) y extrae las matrices Jacobianas LTV (A, B)
        a lo largo del horizonte N.
        """
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

    def _cbf_constraint_multistep(self, u_flat, current_state, obstacle_pos, gamma=GAMMA_CBF):
        """
        Aplica la restricción CBF en tiempo discreto: h(x_{k+1}) - (1 - gamma) * h(x_k) >= 0.
        Se activa únicamente cuando existe un obstáculo detectado.
        """
        if obstacle_pos is None:
            return np.array([0.0])

        u = u_flat.reshape((self.N, 2))
        x_obs, y_obs = obstacle_pos[0], obstacle_pos[1]
        
        cbf_violations = []
        x_curr = current_state.copy()

        for k in range(self.N):
            x_next = KinematicBicycleModel.step(x_curr, u[k])

            v_k = max(x_curr[2], 0.1)
            R_margin = 2.5 + 0.3 * v_k

            h_curr = (x_curr[0] - x_obs)**2 + (x_curr[1] - y_obs)**2 - R_margin**2
            h_next = (x_next[0] - x_obs)**2 + (x_next[1] - y_obs)**2 - R_margin**2
            
            cbf_violations.append((h_next - h_curr) + gamma * h_curr)
            x_curr = x_next

        return np.array(cbf_violations)

    def _cost_function(self, u_flat, current_state, reference_trajectory, u_warm):
        """
        Calcula la función de costo proyectando las desviaciones lineales:
        \delta x_{k+1} = A_k \delta x_k + B_k \delta u_k
        """
        u = u_flat.reshape((self.N, 2))
        cost = 0.0

        # Obtención de la trayectoria nominal y jacobianos en el punto de operación u_warm
        x_bar, A_seq, B_seq = self._predict_trajectory(current_state, u_warm)
        delta_x = np.zeros(4)

        for i in range(self.N):
            delta_u = u[i] - u_warm[i]
            delta_x = np.dot(A_seq[i], delta_x) + np.dot(B_seq[i], delta_u)
            state_pred = x_bar[i + 1] + delta_x

            ref_x, ref_y, ref_v = reference_trajectory[i]

            # Términos de error cuadrático
            cost_pos = ((state_pred[0] - ref_x) ** 2 + (state_pred[1] - ref_y) ** 2) / (2.0 ** 2)
            cost_spd = (state_pred[2] - ref_v) ** 2 / (5.0 ** 2)
            cost_ctrl = (u[i, 0] ** 2) / (0.5 ** 2) + (u[i, 1] ** 2) / (1.0 ** 2)
            
            if i > 0:
                cost_ctrl += ((u[i, 0] - u[i - 1, 0]) ** 2) / (0.1 ** 2)
                cost_ctrl += ((u[i, 1] - u[i - 1, 1]) ** 2) / (2.0 ** 2)

            cost += 0.60 * cost_pos + 0.20 * cost_spd + 0.20 * cost_ctrl

        return cost

    def solve(self, u0_warm, current_state, reference_trajectory, obstacle_pos=None):
        u_warm = u0_warm.reshape((self.N, 2))
        bounds = [(-1.0, 1.0)] * (self.N * 2)

        constraints = []
        if obstacle_pos is not None:
            constraints.append({
                'type': 'ineq',
                'fun': self._cbf_constraint_multistep,
                'args': (current_state, obstacle_pos)
            })

        res = minimize(
            self._cost_function,
            u0_warm,
            args=(current_state, reference_trajectory, u_warm),
            bounds=bounds,
            constraints=constraints,
            method='SLSQP',
            options={'maxiter': 50, 'ftol': 1e-3}
        )

        if res.success:
            return res.x.reshape((self.N, 2)), res.x
        
        return u_warm, u0_warm.flatten()