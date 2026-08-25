import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch 
from stable_baselines3 import PPO
from metadrive.envs.metadrive_env import MetaDriveEnv

env_config = dict(
    use_render=False,
    traffic_density=0.15,
    num_scenarios=50,
    start_seed=42
)

env = MetaDriveEnv(env_config)

model_path = "models_checkpoints/ppo_metadrive.zip"

# Cargar el agente guardado asociándolo al nuevo entorno
print("Cargando el modelo guardado...")
model = PPO.load(model_path, env=env)

# Entrenar por 150,000 pasos adicionales
print("Continuando el entrenamiento...")
model.learn(total_timesteps=150_000, reset_num_timesteps=False)

# Sobrescribir el archivo guardado con las mejoras
model.save("models_checkpoints/ppo_metadrive")
print("Modelo actualizado guardado con éxito.")

env.close()