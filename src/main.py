import os
import numpy as np
import cv2
import imageio

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["CVXPY_ACTIVE_SOLVER"] = "CLARABEL"

from config import N, DT, TOTAL_STEPS, FPS, MPC_SKIP_STEPS, TOLERANCIA_MAX_M, VIDEO_FILENAME
from controllers.nominal_mpc import NominalMPC
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.engine.engine_utils import close_engine, engine_initialized


def ejecutar_simulacion():
    if engine_initialized():
        close_engine()

    num_escenarios = 1
    start_seed = 47
    VIDEO_SKIP = 7

    env = MetaDriveEnv(dict(
        use_render=False,
        num_scenarios=num_escenarios,
        start_seed=start_seed,
        traffic_density=0.2,
        map="OCCO",
        crash_object_done=False,
        out_of_road_done=False
    ))

    mpc = NominalMPC(horizon=N)
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

                # Planificación MPC
                if step % MPC_SKIP_STEPS == 0:
                    ref_trajectory = []
                    target_speed = 8.0
                    for i in range(N):
                        target_s = current_s + target_speed * (i + 1) * DT
                        ref_x, ref_y = lane.position(target_s, 0)
                        ref_trajectory.append((ref_x, ref_y, target_speed))

                    heading_vec = np.array([np.cos(state_real[3]), np.sin(state_real[3])])
                    obstacle_pos = None
                    vehicles = env.engine.traffic_manager.vehicles
                    min_dist = 25.0
                    
                    # Detectar obstáculo más cercano al frente
                    for v in vehicles:
                        if v != vehicle:
                            rel_pos = v.position - vehicle.position
                            dist_adelante = np.dot(rel_pos, heading_vec)
                            dist_total = np.linalg.norm(rel_pos)
                            
                            if 0 < dist_adelante < min_dist and dist_total < min_dist:
                                min_dist = dist_adelante
                                obstacle_pos = np.array([v.position[0], v.position[1]])
                    
                    # Evaluación del freno de emergencia
                    freno_emergencia = False
                    if obstacle_pos is not None:
                        dist_obs = np.linalg.norm(state_real[:2] - obstacle_pos)
                        v_curr = max(state_real[2], 0.0)
                        dist_critica = 5.0 + 0.8 * v_curr
                    
                        if dist_obs < dist_critica:
                            freno_emergencia = True
                    
                    # Resolver MPC con CBF activa
                    u_nom_seq, u0_warm_flat = mpc.solve(
                        u0_warm, 
                        state_real, 
                        ref_trajectory, 
                        obstacle_pos=obstacle_pos
                    )

                    if freno_emergencia:
                        u_nom_first = np.array([u_nom_seq[0, 0], -1.0])
                    else:
                        u_nom_first = u_nom_seq[0]
                    
                    u0_warm = np.roll(u0_warm_flat, -2)
                    u0_warm[-2:] = u0_warm[-4:-2]

                obs, reward, terminated, truncated, info = env.step(u_nom_first)
                pasos_completados += 1

                # === CAPTURA Y ANOTACIÓN DE VIDEO ===
                if step % VIDEO_SKIP == 0:
                    screen_w, screen_h = 600, 600
                    scaling = 5  # píxeles por metro

                    frame = env.render(
                        mode="topdown",
                        window=False,
                        screen_size=(screen_w, screen_h),
                        camera_position=vehicle.position,
                        target_vehicle_heading_up=True,
                        scaling=scaling,
                        text={
                            "seed": seed,
                            "step": step,
                            "speed_kmh": round(state_real[2] * 3.6, 1)
                        }
                    )

                    # Dibujar círculos de barrera si se detecta un obstáculo
                    if obstacle_pos is not None:
                        v_curr = max(state_real[2], 0.0)
                        r_cbf_m = 10.0 + 0.8 * v_curr      # Margen preventivo CBF
                        r_aeb_m = 5.0 + 0.8 * v_curr       # Margen crítico AEB

                        # Transformar posición del obstáculo a píxeles de la cámara
                        rel_pos = obstacle_pos - state_real[:2]
                        theta = state_real[3]

                        # Proyección en ejes local del vehículo (arriba/derecha)
                        d_fwd = rel_pos[0] * np.cos(theta) + rel_pos[1] * np.sin(theta)
                        d_right = rel_pos[0] * np.sin(theta) - rel_pos[1] * np.cos(theta)

                        center_x = int(screen_w / 2 + d_right * scaling)
                        center_y = int(screen_h / 2 - d_fwd * scaling)

                        r_cbf_px = int(r_cbf_m * scaling)
                        r_aeb_px = int(r_aeb_m * scaling)

                        # Formato de color RGB para imageio
                        # Círculo Exterior (CBF Preventivo): Amarillo (255, 255, 0)
                        cv2.circle(frame, (center_x, center_y), r_cbf_px, (255, 255, 0), 2)
                        
                        # Círculo Interior (AEB Freno Crítico): Rojo (255, 0, 0)
                        cv2.circle(frame, (center_x, center_y), r_aeb_px, (255, 0, 0), 2)

                    writer.append_data(frame)

                    if step % int(5 * VIDEO_SKIP) == 0:
                        print(f"{step} / 2000 -- VELOCIDAD: {max(state_real[2], 0.0):.2f} m/s")

                # Registro de desviación
                ancho_carril = lane.width
                limite_borde = ancho_carril / 2.0

                if abs(lat_error) > limite_borde:
                    exceso_salida = max(exceso_salida, abs(lat_error) - limite_borde)

                if terminated or truncated:
                    if info.get("arrive_dest", False):
                        exito = True
                    break

            resultados.append({
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
            print("\n" + "=" * 92)
            print(f"{'SEMILLA':<8} | {'ÉXITO':<6} | {'ERR LAT PROM (m)':<17} | {'ERR LAT MÁX (m)':<16} | {'EXCESO SALIDA (m)':<18} | {'PASOS':<6}")
            print("=" * 92)
            for r in resultados:
                print(f"{r['Seed']:<8} | {r['Éxito']:<6} | {r['Err. Lat. Promedio (m)']:<17.3f} | {r['Err. Lat. Máximo (m)']:<16.3f} | {r['Exceso Salida (m)']:<18.3f} | {r['Pasos']:<6}")
            print("=" * 92)


if __name__ == "__main__":
    ejecutar_simulacion()