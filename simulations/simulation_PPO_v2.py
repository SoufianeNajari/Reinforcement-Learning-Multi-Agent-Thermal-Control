import os
import sys
import numpy as np
import pandas as pd
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.environment import BuildingEnv
import config as cfg

config = {
    "model_name": "PPO_v2",
    "total_timesteps": 1e6,
    #"time_steps_eval": 2880
}


# 1. Preparation of the environment for SB3
raw_env = BuildingEnv(cfg.BUILDING_CONFIG, render_mode=None)

# On transforme l'env PettingZoo en un format compréhensible par l'IA (Vectorized Env)
env_train = ss.pettingzoo_env_to_vec_env_v1(raw_env)
env_train = ss.concat_vec_envs_v1(env_train, num_vec_envs=1, num_cpus=0, base_class="stable_baselines3")
env_train = VecMonitor(env_train)

# 2. Model (PPO)
model = PPO(
    "MlpPolicy",
    env_train, 
    verbose=1,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,    
    ent_coef=0.01,    # Force l'IA à explorer (évite le std=0)
    device="auto"
)

# 3. Training
print(f"--- Début de l'entraînement de {config['model_name']} ---")
model.learn(total_timesteps=config["total_timesteps"], progress_bar=True)
model.save(f"models/{config['model_name']}")
