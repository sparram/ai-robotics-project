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

# EJECUTAR SIMULACIÓN
def run(control_type="MPC-CBF"):
    if engine_initialized():
        close_engine()

    num_escenarios = 1
    start_seed = 37
    VIDEO_SKIP = 5

    env = MetaDriveEnv(dict(
        use_render=False,
        num_scenarios=num_escenarios,
        start_seed=start_seed,
        traffic_density=0.2,
        map="OCCO",
        crash_object_done=False,
        out_of_road_done=False
    ))

    # Seleccionar controlador a utilizar
    if control_type == "MPC-CBF":
        mpc_cbf = MPC_CBF(horizon=N)
    elif control_type == "RL":
        rl_agent = RLController("models_checkpoints/ppo_metadrive.zip")
    else:
        raise ValueError(f"Error: Selecciona un controlador válido: MPC-CBF o RL")

    writer = imageio.get_writer(VIDEO_FILENAME, fps=FPS)
    resultados = []

    try:
        # EJECUTAR TODOS LOS ESCENARIOS
        for seed in range(start_seed, start_seed + num_escenarios):
            obs, info = env.reset(seed=seed)
            u0_warm = np.zeros(N * 2)
            u_nom_first = np.array([0.0, 0.0])

            errores_laterales = []
            exito = False
            pasos_completados = 0
            exceso_salida = 0.0

            # EJECUTAR LOS PASOS DE LA SIMULACIÓN
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

                # CODIGO DE CONTROL
                # CONTROL Rl:
                if control_type == "RL":
                    u_nom_first = rl_agent.solve(obs) # Llama el modelo RL

                    obstacle_pos = None
                    v_curr = max(state_real[2], 0.0)  # Velocidad actual en m/s
                
                    # Limitador de velocidad en rotondas (Precaución)
                    v_max_rotonda = 4.0
                    if v_curr > v_max_rotonda:
                        u_nom_first[1] = min(u_nom_first[1], -0.3) # Desacelerar en caso de superar la v_max
                
                    # MECANISMO AEB : FRENADO DE PRECAUCIÓN
                    vehicles = env.engine.traffic_manager.vehicles
                    theta = state_real[3]
                    dist_critica = 10.0 + 0.5 * v_curr  # Distancia de frenado dinámica
                
                    for v in vehicles:
                        if v != vehicle:
                            rel_pos = v.position - vehicle.position
                            
                            # Proyección longitudinal y lateral de distancias al obstaculo
                            d_fwd = rel_pos[0] * np.cos(theta) + rel_pos[1] * np.sin(theta)
                            d_right = rel_pos[0] * np.sin(theta) - rel_pos[1] * np.cos(theta)
                
                            # CASO A: Peligro inminente -> Freno total (-1.0)
                            if 0 < d_fwd < dist_critica and abs(d_right) < 1.5:
                                u_nom_first[1] = -1.0
                                obstacle_pos = np.array([v.position[0], v.position[1]])  # Para el dibujo visual
                                break
                
                            # CASO B: Reanudación / Distancia media (5 a 10 metros) -> Aceleración limitada
                            elif 0 < d_fwd < 10.0 and abs(d_right) < 1.5:
                                u_nom_first[1] = min(u_nom_first[1], 0.05)  # Avanzar muy despacio
                                obstacle_pos = np.array([v.position[0], v.position[1]])  # Para el dibujo visual
                                break
                            else:
                                pass
                # CONTROL MPC-CBF
                elif control_type == "MPC-CBF":
                    if step % MPC_SKIP_STEPS == 0:
                        # OBTENER TRAYECTORIA DE REFERENCIA
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
                        
                        # Evaluación del freno de emergencia AEB
                        freno_emergencia = False
                        if obstacle_pos is not None:
                            dist_obs = np.linalg.norm(state_real[:2] - obstacle_pos)
                            v_curr = max(state_real[2], 0.0)
                            dist_critica = 10.0 + 0.8 * v_curr
                        
                            if dist_obs < dist_critica:
                                freno_emergencia = True
                        
                        # Resolver MPC con CBF activa
                        u_nom_seq, u0_warm_flat = mpc_cbf.solve(
                            u0_warm, 
                            state_real, 
                            ref_trajectory, 
                            obstacle_pos=obstacle_pos
                        )

                        if freno_emergencia:
                            u_nom_first = np.array([u_nom_seq[0, 0], -1.0])
                        else:
                            u_nom_first = u_nom_seq[0]

                        # Actualizar el u0 base para la siguiente optimización
                        u0_warm = np.roll(u0_warm_flat, -2)
                        u0_warm[-2:] = u0_warm[-4:-2]

                # Aplicación de la acción en el entorno
                obs, reward, terminated, truncated, info = env.step(u_nom_first)
                pasos_completados += 1

                # PARTE DE VISUALIZACIÓN
                # CREACIÓN DEL VIDEO
                if step % VIDEO_SKIP == 0:
                    screen_w, screen_h = 608, 608
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
                            "speed_kmh": round(state_real[2] * 3.6, 1),
                            "mode": control_type
                        }
                    )

                    # Dibujar círculos si hay obstáculo detectado
                    if control_type == "MPC-CBF" and obstacle_pos is not None:
                        v_curr = max(state_real[2], 0.0)
                        r_cbf_m = 2.5 + 0.3 * v_curr
                        r_aeb_m = 10.0 + 0.8 * v_curr

                        rel_pos = obstacle_pos - state_real[:2]
                        theta = state_real[3]

                        d_fwd = rel_pos[0] * np.cos(theta) + rel_pos[1] * np.sin(theta)
                        d_right = rel_pos[0] * np.sin(theta) - rel_pos[1] * np.cos(theta)

                        center_x = int(screen_w / 2 + d_right * scaling)
                        center_y = int(screen_h / 2 - d_fwd * scaling)

                        r_cbf_px = int(r_cbf_m * scaling)
                        r_aeb_px = int(r_aeb_m * scaling)

                        cv2.circle(frame, (center_x, center_y), r_cbf_px, (255, 255, 0), 2)
                        cv2.circle(frame, (center_x, center_y), r_aeb_px, (255, 0, 0), 2)
                        
                    elif control_type == "RL" and obstacle_pos is not None:
                        v_curr = max(state_real[2], 0.0)
                        r_aeb_m = 6.0 + 0.6 * v_curr  # Radio dinámico del AEB
                    
                        rel_pos = obstacle_pos - state_real[:2]
                        theta = state_real[3]
                    
                        d_fwd = rel_pos[0] * np.cos(theta) + rel_pos[1] * np.sin(theta)
                        d_right = rel_pos[0] * np.sin(theta) - rel_pos[1] * np.cos(theta)
                    
                        center_x = int(screen_w / 2 + d_right * scaling)
                        center_y = int(screen_h / 2 - d_fwd * scaling)
                    
                        r_aeb_px = int(r_aeb_m * scaling)
                    
                        # Anillo rojo de activación AEB
                        cv2.circle(frame, (center_x, center_y), r_aeb_px, (255, 0, 0), 2)
                    else:
                        pass

                    writer.append_data(frame)

                    if step % int(5 * VIDEO_SKIP) == 0:
                        print(f"[{control_type}] Paso {step} / {TOTAL_STEPS} -- VELOCIDAD: {max(state_real[2], 0.0):.2f} m/s")

                # Guardar las desviaciones y salidas del carril
                ancho_carril = lane.width
                limite_borde = ancho_carril / 2.0

                if abs(lat_error) > limite_borde:
                    exceso_salida = max(exceso_salida, abs(lat_error) - limite_borde)

                if terminated or truncated:
                    if info.get("arrive_dest", False):
                        exito = True
                    break

            # Guardar resultados
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

        # Imprimir resultados
        if resultados:
            print("\n" + "=" * 98)
            print(f"{'MODO':<6} | {'SEMILLA':<8} | {'ÉXITO':<6} | {'ERR LAT PROM (m)':<17} | {'ERR LAT MÁX (m)':<16} | {'EXCESO SALIDA (m)':<18} | {'PASOS':<6}")
            print("=" * 98)
            for r in resultados:
                print(f"{r['Controlador']:<6} | {r['Seed']:<8} | {r['Éxito']:<6} | {r['Err. Lat. Promedio (m)']:<17.3f} | {r['Err. Lat. Máximo (m)']:<16.3f} | {r['Exceso Salida (m)']:<18.3f} | {r['Pasos']:<6}")
            print("=" * 98)


if __name__ == "__main__":
    run(control_type="MPC-CBF") # control_type: MPC-CBF o RL