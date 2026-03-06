import os
import sys
import numpy as np
import pandas as pd
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import EvalCallback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.environment import BuildingEnv
import config as cfg

config = {
    "model_name": "PPO_v5_HVAC",
    "total_timesteps": 3e6,
    "log_dir": "logs/"
}

os.makedirs(config["log_dir"], exist_ok=True)
os.makedirs("models", exist_ok=True)

# 1. Preparation of the environment for SB3
raw_env = BuildingEnv(cfg.BUILDING_CONFIG, render_mode=None)

# On transforme l'env PettingZoo en un format compréhensible par l'IA (Vectorized Env)
env_train = ss.pettingzoo_env_to_vec_env_v1(raw_env)
env_train = ss.concat_vec_envs_v1(env_train, num_vec_envs=1, num_cpus=0, base_class="stable_baselines3")

monitor_path = os.path.join(config["log_dir"], config["model_name"])
env_train = VecMonitor(env_train, filename=monitor_path)

# 2. Model (PPO)
model = PPO(
    "MlpPolicy",
    env_train, 
    verbose=1,
    learning_rate=0.0001,      # On divise par 3 pour plus de stabilité
    n_steps=4096,              # On double pour une meilleure estimation
    batch_size=128,            # On augmente pour lisser les gradients
    ent_coef=0.01,             
    clip_range=0.1,            # On réduit (defaut 0.2) pour des mises à jour plus prudentes
    gae_lambda=0.95,           # Aide à réduire la variance des avantages
    n_epochs=10,               
    device="auto"
)

eval_callback = EvalCallback(
    env_train, 
    best_model_save_path="./models/best/",
    log_path="./logs/results/", 
    eval_freq=20000, # Évalue tous les 10k pas
    deterministic=True, 
    render=False
)

# 3. Training
print(f"--- Début de l'entraînement de {config['model_name']} ---")
model.learn(total_timesteps=config["total_timesteps"], callback=eval_callback, progress_bar=True)
model.save(f"models/{config['model_name']}")


