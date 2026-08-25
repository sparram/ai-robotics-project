import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch 
from stable_baselines3 import PPO
from metadrive.envs.metadrive_env import MetaDriveEnv

# Crear entorno de MetaDrive y su configuración
env_config = dict(
    use_render=False,
    traffic_density=0.15,
    num_scenarios=50,
    start_seed=42
)

env = MetaDriveEnv(env_config)
model_path = "src/models_checkpoints/ppo_metadrive.zip"

# Cargar el checkpoint del modelo
# Si no se encuentra, se inicializa un modelo desde cero
try:
    print("Cargando el modelo guardado...")
    model = PPO.load(model_path, env=env)
    print("Continuando el entrenamiento...")
except Exception as e:
    print(f"El modelo no existe en la ruta. Inicializando un nuevo modelo...")
    model = PPO("MlpPolicy", env, verbose=1)

# Entrenar modelo por 150000 pasos
model.learn(total_timesteps=150_000, reset_num_timesteps=False)
model.save("src/models_checkpoints/ppo_metadrive")
print("Modelo actualizado / guardado con éxito.")

env.close()