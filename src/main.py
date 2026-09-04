import os
import cv2
import imageio
import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from config import N, TOTAL_STEPS, FPS, MPC_SKIP_STEPS
from controllers.mpc_cbf_controller import MPC_CBF
from controllers.rl_controller import RLController
from utils.metrics import EpisodeMetrics
from utils.visualization import render_frame

from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.engine.engine_utils import close_engine, engine_initialized

def run(control_type="MPC-CBF"):
    if engine_initialized():
        close_engine()

    num_escenarios = 10
    start_seed = 37
    VIDEO_SKIP = 5

    os.makedirs("media", exist_ok=True)

    env = MetaDriveEnv(dict(
        use_render=False,
        num_scenarios=num_escenarios,
        start_seed=start_seed,
        traffic_density=0.15,
        map="CCCCC",
        crash_object_done=False,
        out_of_road_done=False
    ))

    # Initiate the chosen controller
    if control_type == "MPC-CBF":
        controller = MPC_CBF(horizon=N)
    elif control_type == "RL":
        controller = RLController("models_checkpoints/ppo_metadrive.zip")
    else:
        raise ValueError("Choose a valid controller: MPC-CBF, RL")

    results = []

    try:
        for seed in range(start_seed, start_seed + num_escenarios):
            video_filename = os.path.join("media", f"video_{control_type}_seed_{seed}.mp4")
            writer = imageio.get_writer(video_filename, fps=FPS)

            metrics = EpisodeMetrics()

            try:
                obs, info = env.reset(seed=seed)
                u0_warm = np.zeros(N * 2)
                u_action = np.array([0.0, 0.0])
                obstacles_list = []

                for step in range(TOTAL_STEPS):
                    # Get vehicle and it's variables
                    vehicle = env.agent
                    state_real = np.array([
                        vehicle.position[0],
                        vehicle.position[1],
                        vehicle.speed_km_h / 3.6,
                        vehicle.heading_theta
                    ])

                    # Execute the control
                    if control_type == "RL":
                        u_action = controller.get_action(obs, env, state_real)
                    elif control_type == "MPC-CBF":
                        if step % MPC_SKIP_STEPS == 0:
                            u_action, u0_warm, obstacles_list = controller.get_action(env, state_real, u0_warm)

                    # Step the control in the MetaDrive Environment
                    obs, reward, terminated, truncated, info = env.step(u_action)

                    # Compute errors and metrics
                    metrics.update(env, vehicle, state_real, u_action, info, terminated, truncated)

                    # Render and save video frame
                    if step % VIDEO_SKIP == 0:
                        frame = render_frame(env, vehicle, state_real, control_type, step, seed, u_action, obstacles_list)
                        writer.append_data(frame)

                    if terminated or truncated:
                        break

                results.append(metrics.get_summary(control_type, seed))

            finally:
                writer.close()

    finally:
        cv2.destroyAllWindows()
        env.close()

        if results:
            print("\n" + "=" * 115)
            print(f"{'MODO':<8} | {'SEMILLA':<7} | {'ÉXITO':<6} | {'ERR LAT PROM':<12} | {'EXCESO SALIDA':<14} | {'DIST MÍN':<10} | {'JERK PROM':<10} | {'STEER RATE':<10} | {'PASOS':<6}")
            print("=" * 115)
            for r in results:
                print(f"{r['Controlador']:<8} | {r['Seed']:<7} | {r['Éxito']:<6} | {r['Err. Lat. Prom (m)']:<12.3f} | {r['Exceso Salida (m)']:<14.3f} | {r['Dist. Mín Obs (m)']:<10.3f} | {r['Jerk Prom (1/s)']:<10.3f} | {r['Steer Rate Prom (rad/s)']:<10.3f} | {r['Pasos']:<6}")
            print("=" * 115)

if __name__ == "__main__":
    run(control_type="MPC-CBF") # Change control_type between MPC-CBF and RL