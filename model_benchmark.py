import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import supersuit as ss
from stable_baselines3 import PPO
from core.environment import BuildingEnv
import config as cfg

# 1. CONFIGURATION

benchmark_config = {
    "model_name": "PPO_v5_HVAC", 
    #"model_name": "best_model",        
    "time_steps_eval": 180,    
}

# 2. PRÉPARATION
print("--- Préparation de l'environnement ---")
raw_env = BuildingEnv(cfg.BUILDING_CONFIG.copy(), render_mode=None, random_start=False)
raw_env.target_temp = cfg.TARGET_TEMP

env = ss.pettingzoo_env_to_vec_env_v1(raw_env)
env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=0, base_class="stable_baselines3")

# 3. CHARGEMENT
model_path = f"models/{benchmark_config['model_name']}"
try:
    model = PPO.load(model_path)
    print(f"Modèle chargé : {model_path}")
except FileNotFoundError:
    print(f"ERREUR: Fichier introuvable {model_path}")
    exit()

# 4. SIMULATION
print(f"--- Lancement simulation ({benchmark_config['time_steps_eval']} min) ---")
raw_env.default_t_ext = cfg.TRAIN_T_EXT
obs = env.reset() 
data = []

for step in range(benchmark_config["time_steps_eval"]):
    actions, _ = model.predict(obs, deterministic=True)
    next_obs, rewards, dones, infos = env.step(actions)
    
    row = {
        "step": step,
        "target": cfg.TARGET_TEMP,
        "price": obs[0][4] 
    }
    
    for i in range(3):
        agent_name = f"zone_{i}"
        
        # Récupération de l'erreur (ex: -2.0)
        error_val = float(obs[i][0])
        
        # Reconstruction de la vraie température (ex: -2.0 + 21.0 = 19.0)
        real_temp = error_val + cfg.TARGET_TEMP
        
        row[f"temp_{agent_name}"] = real_temp
        row[f"act_{agent_name}"] = float(actions[i][0])

    data.append(row)
    obs = next_obs


# 5. RÉSULTATS
df = pd.DataFrame(data)
os.makedirs("results", exist_ok=True)
csv_filename = f"results/data_model_{benchmark_config['model_name']}.csv"
df.to_csv(csv_filename, index=False)
print(f"Données sauvegardées : {csv_filename}")

print("Génération du graphique...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

for col in [c for c in df.columns if c.startswith("temp_")]:
    ax1.plot(df["step"], df[col], label=col.replace("temp_", "Zone "))
ax1.axhline(y=cfg.TARGET_TEMP, color='k', linestyle='--', label="Cible")
ax1.set_ylabel("Température (°C)")
ax1.set_title(f"Test à T_ext = {cfg.TRAIN_T_EXT}°C")
ax1.legend()
ax1.grid(True, alpha=0.3)

for col in [c for c in df.columns if c.startswith("act_")]:
    ax2.plot(df["step"], df[col], label=col.replace("act_", "Zone "), alpha=0.8)
ax2.set_ylabel("Chauffage")
ax2.set_xlabel("Minutes")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()