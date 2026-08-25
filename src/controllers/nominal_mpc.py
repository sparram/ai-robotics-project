import numpy as np
from scipy.optimize import minimize
from config import N, DT
from models.kinematic import KinematicBicycleModel

class NominalMPC:
    def __init__(self, horizon=N):
        self.N = horizon

    def _predict_trajectory(self, current_state, u_seq):
        x_bar = np.zeros((self.N + 1, 4))
        x_bar[0] = current_state
        A_seq = []
        B_seq = []

        for i in range(self.N):
            A = KinematicBicycleModel.jacobian_x(x_bar[i], u_seq[i])
            B = KinematicBicycleModel.jacobian_u(x_bar[i], u_seq[i])
            A_seq.append(A)
            B_seq.append(B)
            x_bar[i + 1] = KinematicBicycleModel.step(x_bar[i], u_seq[i])

        return x_bar, A_seq, B_seq

    def _cbf_constraint_multistep(self, u_flat, current_state, obstacle_pos, gamma=0.9):
        """
        Aplica la condicion CBF h(x_{k+1}) - (1 - gamma)*h(x_k) >= 0 
        a lo largo de todo el horizonte N.
        """
        u = u_flat.reshape((self.N, 2))
        x_obs, y_obs = obstacle_pos[0], obstacle_pos[1]
        
        # Margen dinamico adaptado a la velocidad actual
        v_current = max(current_state[2], 1.0)
        R_margin = 2.5 + 0.3 * v_current  

        cbf_violations = []
        x_curr = current_state.copy()

        for k in range(self.N):
            # h(x_k)
            h_curr = (x_curr[0] - x_obs)**2 + (x_curr[1] - y_obs)**2 - R_margin**2
            
            # Siguiente estado x_{k+1}
            x_next = KinematicBicycleModel.step(x_curr, u[k])
            
            # h(x_{k+1})
            h_next = (x_next[0] - x_obs)**2 + (x_next[1] - y_obs)**2 - R_margin**2
            
            # Condicion CBF discreta: h(x_{k+1}) - (1 - gamma)*h(x_k) >= 0
            cbf_violations.append((h_next - h_curr) + gamma * h_curr)
            
            x_curr = x_next

        # Retorna el arreglo de restricciones (SLSQP exige que cada elemento sea >= 0)
        return np.array(cbf_violations)

    def _cost_function(self, u_flat, current_state, reference_trajectory, u_warm):
        u = u_flat.reshape((self.N, 2))
        cost = 0.0

        x_bar, A_seq, B_seq = self._predict_trajectory(current_state, u_warm)
        delta_x = np.zeros(4)

        for i in range(self.N):
            delta_u = u[i] - u_warm[i]
            delta_x = np.dot(A_seq[i], delta_x) + np.dot(B_seq[i], delta_u)
            state_pred = x_bar[i + 1] + delta_x

            ref_x, ref_y, ref_v = reference_trajectory[i]

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
        
        # Si falla la optimizacion, aplicar frenado de emergencia en lugar de seguir u_warm
        u_brake = u_warm.copy()
        u_brake[:, 1] = -1.0  # Deceleracion maxima
        return u_brake, u_brake.flatten()