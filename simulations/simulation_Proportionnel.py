import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.environment import BuildingEnv
import config as cfg

config = {
    "model_name": "model_Proportionnel_climatisation",
    "time_steps": 90,    
    "Kp": 1.0
}
    
env = BuildingEnv(cfg.BUILDING_CONFIG.copy(), render_mode=None, random_start=False)

env.default_t_ext = cfg.TRAIN_T_EXT
observations, _ = env.reset()

data = []

for step in range(config["time_steps"]):
    actions = {}
    row = {"step": step, "target": env.target_temp}
    
    for agent in env.possible_agents:
        erreur = observations[agent][0]
        
        temp_actuelle = erreur + env.target_temp
        
        val_action = np.clip(config["Kp"] * (-erreur), -1.0, 1.0)
        
        actions[agent] = np.array([val_action], dtype=np.float32)
        
        row[f"temp_{agent}"] = temp_actuelle
        row[f"act_{agent}"] = float(val_action)

    observations, _, _, _, _ = env.step(actions, t_ext=cfg.TRAIN_T_EXT)
    data.append(row)

os.makedirs("results", exist_ok=True)
filename = f"results/data_{config['model_name']}.csv"
pd.DataFrame(data).to_csv(filename, index=False)
print(f"Simulation terminée. Résultats dans : {filename}")