import numpy as np
from config import DT

class EpisodeMetrics:
    """Clase para registrar y calcular las métricas de evaluación durante un episodio."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.errores_laterales = []
        self.exceso_salida = 0.0
        self.distancias_obs = []
        self.jerks = []
        self.steer_rates = []
        self.u_prev = None
        self.pasos_completados = 0
        self.exito = False

    def update(self, env, vehicle, state_real, u_action, info, terminated, truncated):
        # 1. Error lateral
        lane = vehicle.navigation.current_lane
        _, lat_error = lane.local_coordinates((state_real[0], state_real[1]))
        self.errores_laterales.append(abs(lat_error))

        # 2. Exceso de salida del carril
        ancho_carril = lane.width
        limite_borde = ancho_carril / 2.0
        if abs(lat_error) > limite_borde:
            self.exceso_salida = max(self.exceso_salida, abs(lat_error) - limite_borde)

        # 3. Distancias a otros vehículos
        vehicles = env.engine.traffic_manager.vehicles
        for v in vehicles:
            if v != vehicle:
                dist = np.linalg.norm(v.position - vehicle.position)
                self.distancias_obs.append(dist)

        # 4. Métricas de suavidad de control (Steer Rate y Jerk)
        if self.u_prev is not None:
            self.steer_rates.append(abs(u_action[0] - self.u_prev[0]) / DT)
            self.jerks.append(abs(u_action[1] - self.u_prev[1]) / DT)
        self.u_prev = u_action.copy()

        self.pasos_completados += 1

        if terminated or truncated:
            if info.get("arrive_dest", False):
                self.exito = True

    def get_summary(self, control_type, seed):
        return {
            "Controlador": control_type,
            "Seed": seed,
            "Éxito": "SÍ" if self.exito else "NO",
            "Err. Lat. Prom (m)": float(np.mean(self.errores_laterales)) if self.errores_laterales else 0.0,
            "Err. Lat. Máx (m)": float(np.max(self.errores_laterales)) if self.errores_laterales else 0.0,
            "Exceso Salida (m)": float(self.exceso_salida),
            "Dist. Mín Obs (m)": float(np.min(self.distancias_obs)) if self.distancias_obs else 99.0,
            "Jerk Prom (1/s)": float(np.mean(self.jerks)) if self.jerks else 0.0,
            "Steer Rate Prom (rad/s)": float(np.mean(self.steer_rates)) if self.steer_rates else 0.0,
            "Pasos": self.pasos_completados
        }