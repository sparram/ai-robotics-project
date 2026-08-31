import os
import csv
import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from config import N, TOTAL_STEPS, MPC_SKIP_STEPS
from controllers.mpc_cbf_controller import MPC_CBF
from controllers.rl_controller import RLController
from utils.metrics import EpisodeMetrics
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.engine.engine_utils import close_engine, engine_initialized


def evaluar_semilla(control_type, controller, seed, env):
    obs, info = env.reset(seed=seed)
    u0_warm = np.zeros(N * 2)
    u_action = np.array([0.0, 0.0])
    metrics = EpisodeMetrics()

    for step in range(TOTAL_STEPS):
        vehicle = env.agent
        state_real = np.array([
            vehicle.position[0],
            vehicle.position[1],
            vehicle.speed_km_h / 3.6,
            vehicle.heading_theta
        ])

        # 1. Delegar obtención de control
        if control_type == "RL":
            u_action = controller.get_action(obs, env, state_real)
        elif control_type == "MPC-CBF":
            if step % MPC_SKIP_STEPS == 0:
                u_action, u0_warm, _ = controller.get_action(env, state_real, u0_warm)

        # 2. Avanzar entorno
        obs, reward, terminated, truncated, info = env.step(u_action)

        # 3. Registrar métricas
        metrics.update(env, vehicle, state_real, u_action, info, terminated, truncated)

        if terminated or truncated:
            break

    return metrics.get_summary(control_type, seed)


def ejecutar_benchmark(num_escenarios=10, start_seed=37):
    if engine_initialized():
        close_engine()

    env = MetaDriveEnv(dict(
        use_render=False,
        num_scenarios=num_escenarios,
        start_seed=start_seed,
        traffic_density=0.15,
        map="CCCCC",
        crash_object_done=False,
        out_of_road_done=False
    ))

    # Instanciar ambos controladores una sola vez
    controladores = {
        "MPC-CBF": MPC_CBF(horizon=N),
        "RL": RLController("models_checkpoints/ppo_metadrive.zip")
    }

    resumen_global = []
    detalles_semillas = []

    try:
        for control_type, controller in controladores.items():
            print(f"\n[BENCHMARK] Evaluando {control_type} en {num_escenarios} escenarios...")

            resultados_lote = []
            for seed in range(start_seed, start_seed + num_escenarios):
                res = evaluar_semilla(control_type, controller, seed, env)
                resultados_lote.append(res)
                detalles_semillas.append(res)

                print(f"  -> [{control_type}] Semilla {seed}: Éxito={res['Éxito']} | ErrLatProm={res['Err. Lat. Prom (m)']:.2f}m")

            exitos = sum(1 for r in resultados_lote if r["Éxito"] == "SÍ")
            resumen_global.append({
                "Controlador": control_type,
                "Éxito (%)": (exitos / num_escenarios) * 100.0,
                "Err Lat (m)": float(np.mean([r["Err. Lat. Prom (m)"] for r in resultados_lote])),
                "Salida (m)": float(np.mean([r["Exceso Salida (m)"] for r in resultados_lote])),
                "Dist Mín (m)": float(np.mean([r["Dist. Mín Obs (m)"] for r in resultados_lote])),
                "Jerk Prom": float(np.mean([r["Jerk Prom (1/s)"] for r in resultados_lote])),
                "Steer Rate": float(np.mean([r["Steer Rate Prom (rad/s)"] for r in resultados_lote]))
            })

    finally:
        env.close()

    print("\n" + "=" * 90)
    print(f"{'CONTROLADOR':<12} | {'ÉXITO (%)':<10} | {'ERR LAT (m)':<12} | {'SALIDA (m)':<10} | {'DIST MÍN (m)':<12} | {'JERK PROM':<10} | {'STEER RATE':<10}")
    print("=" * 90)
    for g in resumen_global:
        print(f"{g['Controlador']:<12} | {g['Éxito (%)']:<10.1f} | {g['Err Lat (m)']:<12.3f} | {g['Salida (m)']:<10.3f} | {g['Dist Mín (m)']:<12.3f} | {g['Jerk Prom']:<10.3f} | {g['Steer Rate']:<10.3f}")
    print("=" * 90)

    # Guardar CSV resumen
    with open("benchmark_resumen.csv", mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=resumen_global[0].keys())
        writer.writeheader()
        writer.writerows(resumen_global)

    # Guardar CSV detalles
    with open("benchmark_detalles.csv", mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=detalles_semillas[0].keys())
        writer.writeheader()
        writer.writerows(detalles_semillas)

    print("\n¡Resultados guardados en 'benchmark_resumen.csv' y 'benchmark_detalles.csv'!")

if __name__ == "__main__":
    ejecutar_benchmark(num_escenarios=10, start_seed=37)