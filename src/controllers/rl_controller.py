import os
import numpy as np
from stable_baselines3 import PPO

class RLController:
    def __init__(self, model_path="models_checkpoints/ppo_metadrive.zip"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found in path: {model_path}.")
        self.model = PPO.load(model_path)

    # Get the action according to the Neural Network
    def solve(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return action

    # Get the model action and filter it with safety measurements
    def get_action(self, obs, env, state_real):
        u_action, _ = self.model.predict(obs, deterministic=True)
        u_action = u_action.astype(np.float64)

        v_curr = max(state_real[2], 0.0)

        # Speed moderation: Progressive deacceleration
        if v_curr > 4.0:
            u_action[1] = min(u_action[1], -0.3)

        # Scan for nearby obstacles and Emergency Braking System
        vehicle = env.agent
        lane = vehicle.navigation.current_lane
        vehicles = env.engine.traffic_manager.vehicles
        dist_critica = 10.0 + 0.5 * v_curr
        s_ego, lat_ego = lane.local_coordinates(state_real[:2])

        for v in vehicles:
            if v != vehicle:
                s_obs, lat_obs = lane.local_coordinates(v.position)
                d_fwd = s_obs - s_ego
                d_right = lat_obs - lat_ego

                if 0 < d_fwd < dist_critica and abs(d_right) < 1.5:
                    u_action[1] = -1.0
                    break

        return u_action