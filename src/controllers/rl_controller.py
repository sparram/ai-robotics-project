import os
from stable_baselines3 import PPO

class RLController:
    def __init__(self, model_path="models_checkpoints/ppo_metadrive.zip"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encontró el modelo en {model_path}.")
        self.model = PPO.load(model_path)

    # Ejecutar el control
    # Recibe la obsersvación del entorno y retorna el control (steer y aceleración)
    def solve(self, obs):
        action, _states = self.model.predict(obs, deterministic=True)
        return action