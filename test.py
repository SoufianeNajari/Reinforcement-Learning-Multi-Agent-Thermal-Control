import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from core.environment import BuildingEnv

# ==========================================
# 1. CONFIGURATION DU TEST (C'est ici que tu changes les paramètres)
# ==========================================
test_config = {
    "model_path": "models/PPO_v1",  # Nom du fichier sans .zip
    "time_steps_eval": 2880 * 2,    # Exemple : On teste sur 2 jours (48h)
    "t_ext_scenario": 0.0,          # Testons une journée plus froide (0°C)
    "target_temp": 21.0             # La consigne
}

# IMPORTANT : La physique du bâtiment doit rester la même que lors de l'entraînement
building_config = {
    "adj_matrix": [[0, 1, 1], [1, 0, 0], [1, 0, 0]],
    "expo_ext": [0.0, 1.0, 1.0],
    "t_ext_offset": [0.0, -2.0, 3.0], 
    "start_temp": 19.0,
    "R_val": 0.2,
    "C_val": 1e6,
    "R_inter": 0.5,
    "max_power": 2000,
    "dt": 60,
    # On force max_steps à la durée du test pour éviter que l'env ne reset tout seul
    "max_steps": test_config["time_steps_eval"] 
}

# ==========================================
# 2. CHARGEMENT
# ==========================================
print(f"--- Chargement du modèle depuis {test_config['model_path']} ---")
# On charge le cerveau entraîné
model = PPO.load(test_config["model_path"])

# On crée l'environnement de test (pas besoin de SuperSuit ici pour de l'inférence simple)
env = BuildingEnv(building_config, render_mode=None)
env.target_temp = test_config["target_temp"]

# ==========================================
# 3. BOUCLE DE SIMULATION
# ==========================================
print(f"--- Lancement de la simulation sur {test_config['time_steps_eval']} minutes ---")
obs, _ = env.reset(options={"t_ext": test_config["t_ext_scenario"]})
data = []

for step in range(test_config["time_steps_eval"]):
    actions_dict = {}
    row = {
        "step": step, 
        "target": env.target_temp,
        "price": env.get_price() # On enregistre aussi le prix pour voir la réaction
    }
    
    # Pour chaque zone, on demande au modèle quoi faire
    for agent in env.possible_agents:
        # deterministic=True est important pour le test (pas de hasard, que de la logique)
        action, _ = model.predict(obs[agent], deterministic=True)
        actions_dict[agent] = action
        
        # Enregistrement des données
        row[f"temp_{agent}"] = float(obs[agent][0])
        row[f"act_{agent}"] = float(action[0])

    # On applique les actions
    obs, rewards, _, _, _ = env.step(actions_dict, t_ext=test_config["t_ext_scenario"])
    data.append(row)

# ==========================================
# 4. SAUVEGARDE ET GRAPHIQUE
# ==========================================
df = pd.DataFrame(data)
os.makedirs("results", exist_ok=True)
csv_filename = f"results/test_simulation_{test_config['time_steps_eval']}steps.csv"
df.to_csv(csv_filename, index=False)
print(f"Données sauvegardées dans : {csv_filename}")

# --- Plotting automatique ---
print("Génération du graphique...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Zone de prix élevé (optionnel, pour visualiser)
# On repère où le prix est élevé (> 0.20 par exemple)
high_price_mask = df["price"] > 0.20
ax1.fill_between(df["step"], 15, 25, where=high_price_mask, color='red', alpha=0.1, label="Prix Élevé")

# Températures
for col in [c for c in df.columns if c.startswith("temp_")]:
    ax1.plot(df["step"], df[col], label=col.replace("temp_", "Zone "))
ax1.axhline(y=test_config["target_temp"], color='k', linestyle='--', label="Cible")
ax1.set_ylabel("Température (°C)")
ax1.set_title("Évolution thermique")
ax1.legend(loc="upper right")
ax1.grid(True, alpha=0.3)

# Actions
for col in [c for c in df.columns if c.startswith("act_")]:
    ax2.plot(df["step"], df[col], label=col.replace("act_", "Zone "), alpha=0.8)
ax2.set_ylabel("Chauffage (0 à 1)")
ax2.set_xlabel("Minutes")
ax2.set_title("Actions du PPO")
ax2.legend(loc="upper right")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()