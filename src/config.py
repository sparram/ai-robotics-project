# AJUSTE DE HIPERPARÁMETROS
N = 15                  # Horizonte MPC
DT = 0.1               # Paso de tiempo (s)
L = 3.0                # Wheelbase (m)

MPC_SKIP_STEPS = 1     # Pasos a omitir entre cálculos del MPC
GAMMA_CBF = 0.2        # Gamma para la restricción CBF

TOTAL_STEPS = 2000     # Total pasos de la simulación

# Parámetros para la generación del video
FPS = 15
VIDEO_FILENAME = "media/mpc_cbf_example.mp4"