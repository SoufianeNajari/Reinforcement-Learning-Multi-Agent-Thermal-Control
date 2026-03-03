import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.environment import BuildingEnv
import config as cfg

config = {
    "model_name": "model_PI",
    "time_steps": 120,    
    "Kp": 1.0, 
    "Ki": 0.05,
    "t_ext_scenario": 0.0
}
    
env = BuildingEnv(cfg.BUILDING_CONFIG.copy(), render_mode=None)

env.default_t_ext = config["t_ext_scenario"]
observations, _ = env.reset()

data = []
integral_errors = {agent: 0.0 for agent in env.possible_agents}

for step in range(config["time_steps"]):
    actions = {}
    row = {"step": step, "target": env.target_temp}
    
    for agent in env.possible_agents:
        erreur_brute = observations[agent][0]
        erreur_reelle = -erreur_brute
        
        integral_errors[agent] += erreur_reelle
        
        limite_i = 1.0 / config["Ki"] if config["Ki"] > 0 else 0
        integral_errors[agent] = np.clip(integral_errors[agent], -limite_i, limite_i)
        
        temp_actuelle = erreur_brute + env.target_temp
        
        action_p = config["Kp"] * erreur_reelle
        action_i = config["Ki"] * integral_errors[agent]
        
        val_action = np.clip(action_p + action_i, -1.0, 1.0)
        
        actions[agent] = np.array([val_action], dtype=np.float32)
        
        row[f"temp_{agent}"] = temp_actuelle
        row[f"act_{agent}"] = float(val_action)

    observations, _, _, _, _ = env.step(actions, t_ext=config["t_ext_scenario"])
    data.append(row)

os.makedirs("results", exist_ok=True)
filename = f"results/data_{config['model_name']}.csv"
pd.DataFrame(data).to_csv(filename, index=False)
print(f"Simulation terminée. Résultats dans : {filename}")