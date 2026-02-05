import os
import numpy as np
import pandas as pd
import supersuit as ss
from stable_baselines3 import PPO
from core.environment import BuildingEnv
from stable_baselines3.common.vec_env import VecMonitor 
import config as cfg

config = {
    "model_name": "PPO_v1",
    "total_timesteps": 1e6,
    #"time_steps_eval": 2880
}


# 1. Création et préparation de l'environnement pour SB3
raw_env = BuildingEnv(cfg.BUILDING_CONFIG, render_mode=None)

# On transforme l'env PettingZoo en un format compréhensible par l'IA (Vectorized Env)
env_train = ss.pettingzoo_env_to_vec_env_v1(raw_env)
env_train = ss.concat_vec_envs_v1(env_train, num_vec_envs=1, num_cpus=0, base_class="stable_baselines3")
env_train = VecMonitor(env_train)

# 2. Configuration du "Cerveau" (PPO)
model = PPO(
    "MlpPolicy",
    env_train, 
    verbose=1,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,    
    ent_coef=0.01,    # <--- Force l'IA à explorer (évite le std=0)
    device="auto"
)

# 3. Entraînement
print(f"--- Début de l'entraînement de {config['model_name']} ---")
model.learn(total_timesteps=config["total_timesteps"], progress_bar=True)
model.save(f"models/{config['model_name']}")


"""

# 4. Évaluation et Sauvegarde des résultats
print("--- Génération du CSV de performance ---")
eval_env = BuildingEnv(cfg.BUILDING_CONFIG, render_mode=None)
obs, _ = eval_env.reset()
data = []

for step in range(config["time_steps_eval"]):
    actions_dict = {}
    row = {"step": step, "target": eval_env.target_temp}
    
    for agent in eval_env.possible_agents:
        # L'IA prédit maintenant l'action optimale au lieu d'utiliser Kp
        action, _ = model.predict(obs[agent], deterministic=True)
        actions_dict[agent] = action
        
        row[f"temp_{agent}"] = obs[agent][0]
        row[f"act_{agent}"] = float(action[0])

    obs, rewards, _, _, _ = eval_env.step(actions_dict, t_ext=cfg.TRAIN_T_EXT)
    data.append(row)

os.makedirs("results", exist_ok=True)   
filename = f"results/data_{config['model_name']}.csv"
pd.DataFrame(data).to_csv(filename, index=False)
print(f"Terminé ! Fichier créé : {filename}")

"""