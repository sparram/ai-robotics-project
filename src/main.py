import os
import numpy as np
import cv2
import imageio

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from config import N, DT, TOTAL_STEPS, FPS, MPC_SKIP_STEPS, VIDEO_FILENAME
from controllers.mpc_cbf_controller import MPC_CBF
from controllers.rl_controller import RLController

from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.engine.engine_utils import close_engine, engine_initialized

def run(control_type="MPC-CBF"):
    if engine_initialized():
        close_engine()

    num_escenarios = 1
    start_seed = 37
    VIDEO_SKIP = 3  # Ajustado para mayor fluidez en el video

    # 1. ACTIVAMOS EL TRÁFICO (traffic_density > 0)
    env = MetaDriveEnv(dict(
        use_render=False,
        num_scenarios=num_escenarios,
        start_seed=start_seed,
        traffic_density=0.2,  # <-- Tráfico activo
        map="CCCCC",
        crash_object_done=False,
        out_of_road_done=False
    ))

    # Seleccionar controlador
    if control_type == "MPC-CBF":
        mpc_cbf = MPC_CBF(horizon=N)
    elif control_type == "RL":
        rl_agent = RLController("models_checkpoints/ppo_metadrive.zip")
    else:
        raise ValueError("Selecciona un controlador válido: MPC-CBF o RL")

    writer = imageio.get_writer(VIDEO_FILENAME, fps=FPS)
    resultados = []

    try:
        for seed in range(start_seed, start_seed + num_escenarios):
            obs, info = env.reset(seed=seed)
            u0_warm = np.zeros(N * 2)
            u_nom_first = np.array([0.0, 0.0])

            errores_laterales = []
            exito = False
            pasos_completados = 0
            exceso_salida = 0.0

            for step in range(TOTAL_STEPS):
                vehicle = env.agent
                state_real = np.array([
                    vehicle.position[0],
                    vehicle.position[1],
                    vehicle.speed_km_h / 3.6,
                    vehicle.heading_theta
                ])

                lane = vehicle.navigation.current_lane
                current_s, lat_error = lane.local_coordinates((state_real[0], state_real[1]))
                errores_laterales.append(abs(lat_error))

                # -------------------------------------------------------------
                # MODO RL
                # -------------------------------------------------------------
                if control_type == "RL":
                    u_nom_first = rl_agent.solve(obs)
                    v_curr = max(state_real[2], 0.0)

                    if v_curr > 4.0:
                        u_nom_first[1] = min(u_nom_first[1], -0.3)

                    vehicles = env.engine.traffic_manager.vehicles
                    dist_critica = 10.0 + 0.5 * v_curr
                    s_ego, lat_ego = lane.local_coordinates(state_real[:2])

                    for v in vehicles:
                        if v != vehicle:
                            s_obs, lat_obs = lane.local_coordinates(v.position)
                            d_fwd = s_obs - s_ego
                            d_right = lat_obs - lat_ego

                            if 0 < d_fwd < dist_critica and abs(d_right) < 1.5:
                                u_nom_first[1] = -1.0
                                break

                # -------------------------------------------------------------
                # MODO MPC-CBF CON DETECCIÓN FRENET EN CURVAS
                # -------------------------------------------------------------
                elif control_type == "MPC-CBF":
                    if step % MPC_SKIP_STEPS == 0:
                        # Recolectar Obstáculos en un radio de 18 metros
                        obstacles_list = []
                        vehicles = env.engine.traffic_manager.vehicles
                        for v in vehicles:
                            if v != vehicle:
                                v.set_static(True)  # Mantener congelados para pruebas
                                dist = np.linalg.norm(v.position - vehicle.position)
                                if dist < 18.0:
                                    obstacles_list.append(np.array([v.position[0], v.position[1]]))

                        # Detección curvilínea (Frenet) en el carril
                        lat_offset = 0.0
                        s_ego, lat_ego = lane.local_coordinates(state_real[:2])

                        for obs_pos in obstacles_list:
                            s_obs, lat_obs = lane.local_coordinates(obs_pos)
                            d_fwd = s_obs - s_ego          # Distancia longitudinal siguiendo la curva
                            d_right = lat_obs - lat_ego    # Distancia transversal real dentro del carril

                            # Detectar si hay un obstáculo adelante en la trayectoria del carril (< 18m)
                            if 0.0 < d_fwd < 18.0 and abs(d_right) < 1.8:
                                # Esquivar hacia el lado opuesto del obstáculo
                                side = -1.0 if d_right >= 0 else 1.0
                                lat_offset = side * 2.8  # Desplazamiento lateral de referencia
                                break

                        # Construcción de la trayectoria de referencia con desvío
                        ref_trajectory = []
                        is_curved = abs(lane.heading_theta_at(current_s + 5) - lane.heading_theta_at(current_s)) > 0.1
                        target_speed = 5.0 if is_curved else 8.0

                        for i in range(N):
                            target_s = current_s + target_speed * (i + 1) * DT
                            smooth_factor = min(1.0, (i + 1) / (N * 0.5))
                            current_offset = lat_offset * smooth_factor
                            
                            ref_x, ref_y = lane.position(target_s, current_offset)
                            ref_trajectory.append((ref_x, ref_y, target_speed))

                        u_nom_seq, u0_warm_flat = mpc_cbf.solve(
                            u0_warm,
                            state_real,
                            ref_trajectory,
                            obstacles_list=obstacles_list
                        )

                        u_nom_first = u_nom_seq[0]
                        u0_warm = np.roll(u0_warm_flat, -2)
                        u0_warm[-2:] = u0_warm[-4:-2]

                # Aplicar acción
                obs, reward, terminated, truncated, info = env.step(u_nom_first)
                pasos_completados += 1

                # -------------------------------------------------------------
                # RENDERIZADO Y VISUALIZACIÓN MULTI-OBSTÁCULO (CV2)
                # -------------------------------------------------------------
                if step % VIDEO_SKIP == 0:
                    screen_w, screen_h = 608, 608
                    scaling = 5

                    frame = env.render(
                        mode="topdown",
                        window=False,
                        screen_size=(screen_w, screen_h),
                        camera_position=vehicle.position,
                        target_agent_heading_up=True,
                        scaling=scaling,
                        text={
                            "seed": seed,
                            "step": step,
                            "speed_kmh": round(state_real[2] * 3.6, 1),
                            "mode": control_type,
                            "obs_count": len(obstacles_list) if control_type == "MPC-CBF" else 0,
                            "steer": round(u_nom_first[0], 2),
                            "accel": round(u_nom_first[1], 2)
                        }
                    )

                    car_cx, car_cy = screen_w // 2, screen_h // 2

                    # Vector de control
                    u_steer, u_acc = u_nom_first[0], u_nom_first[1]
                    scale_vec = 45.0
                    end_x = int(car_cx - u_steer * scale_vec)
                    end_y = int(car_cy - u_acc * scale_vec)
                    color_vector = (0, 255, 0) if u_acc >= 0 else (0, 0, 255)

                    cv2.arrowedLine(frame, (car_cx, car_cy), (end_x, end_y), color_vector, 3, tipLength=0.35)

                    # Anillos de seguridad CBF
                    if control_type == "MPC-CBF" and obstacles_list:
                        v_curr = max(state_real[2], 0.0)
                        r_cbf_m = 1.4 + 0.2 * v_curr
                        theta = state_real[3]

                        for obs_pos in obstacles_list:
                            rel_pos = obs_pos - state_real[:2]
                            d_fwd_cam = rel_pos[0] * np.cos(theta) + rel_pos[1] * np.sin(theta)
                            d_right_cam = rel_pos[0] * np.sin(theta) - rel_pos[1] * np.cos(theta)

                            center_x = int(screen_w / 2 + d_right_cam * scaling)
                            center_y = int(screen_h / 2 - d_fwd_cam * scaling)

                            cv2.circle(frame, (center_x, center_y), int(r_cbf_m * scaling), (255, 255, 0), 2)

                    writer.append_data(frame)

                # Métricas de salida de carril
                ancho_carril = lane.width
                limite_borde = ancho_carril / 2.0
                if abs(lat_error) > limite_borde:
                    exceso_salida = max(exceso_salida, abs(lat_error) - limite_borde)

                if terminated or truncated:
                    if info.get("arrive_dest", False):
                        exito = True
                    break

            resultados.append({
                "Controlador": control_type,
                "Seed": seed,
                "Éxito": "SÍ" if exito else "NO",
                "Err. Lat. Promedio (m)": np.mean(errores_laterales) if errores_laterales else 0.0,
                "Err. Lat. Máximo (m)": np.max(errores_laterales) if errores_laterales else 0.0,
                "Exceso Salida (m)": exceso_salida,
                "Pasos": pasos_completados
            })

    finally:
        writer.close()
        cv2.destroyAllWindows()
        env.close()

        if resultados:
            print("\n" + "=" * 98)
            print(f"{'MODO':<6} | {'SEMILLA':<8} | {'ÉXITO':<6} | {'ERR LAT PROM (m)':<17} | {'ERR LAT MÁX (m)':<16} | {'EXCESO SALIDA (m)':<18} | {'PASOS':<6}")
            print("=" * 98)
            for r in resultados:
                print(f"{r['Controlador']:<6} | {r['Seed']:<8} | {r['Éxito']:<6} | {r['Err. Lat. Promedio (m)']:<17.3f} | {r['Err. Lat. Máximo (m)']:<16.3f} | {r['Exceso Salida (m)']:<18.3f} | {r['Pasos']:<6}")
            print("=" * 98)

if __name__ == "__main__":
    run(control_type="MPC-CBF")