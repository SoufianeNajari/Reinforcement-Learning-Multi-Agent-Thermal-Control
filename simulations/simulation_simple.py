import numpy as np
import pandas as pd
import os
from core.environment import BuildingEnv



config = {
    "model_name": "PID_Basic",
    "nb_zones": 3,
    "time_steps": 120,    
    "Kp": 5.0, 
    "t_ext_scenario": -160.0
}
    
env = BuildingEnv(nb_zones=config["nb_zones"])
observations, _ = env.reset()

data = []

for step in range(config["time_steps"]):
    actions = {}
    row = {"step": step, "target": env.target_temp}
    
    for agent in env.agents:
        temp_actuelle = observations[agent][0]
        val_action = np.clip(config["Kp"] * (env.target_temp - temp_actuelle), -1.0, 1.0)
        actions[agent] = np.array([val_action], dtype=np.float32)
        
        row[f"temp_{agent}"] = temp_actuelle
        row[f"act_{agent}"] = val_action

    observations, _, _, _, _ = env.step(actions, t_ext=config["t_ext_scenario"])
    data.append(row)

# Sauvegarde avec le nom du modèle
os.makedirs("results", exist_ok=True)
filename = f"results/data_{config['model_name']}.csv"
pd.DataFrame(data).to_csv(filename, index=False)
print(f"Simulation terminée. Résultats dans : {filename}")