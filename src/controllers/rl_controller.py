import os
from stable_baselines3 import PPO

class RLController:
    def __init__(self, model_path="models_checkpoints/ppo_metadrive.zip"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encontró el modelo en {model_path}. Entrénalo primero.")
        self.model = PPO.load(model_path)

    def solve(self, obs):
        """
        Recibe la observación del entorno (state/vector) 
        y retorna la acción predicha por la red [steering, acceleration].
        """
        action, _states = self.model.predict(obs, deterministic=True)
        return action