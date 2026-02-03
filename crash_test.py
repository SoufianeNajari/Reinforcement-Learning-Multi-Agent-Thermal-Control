import os
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor 
from core.environment import BuildingEnv

# --- CONFIGURATION OPTIMISÉE ---
config = {
    "model_name": "PPO_Fix_v2", # On change de nom pour pas mélanger
    "nb_zones": 3,
    "total_timesteps": 500_000, # 500k suffisent largement pour ce problème
    "learning_rate": 0.0003,
    "n_steps": 2048,
    "batch_size": 64,
}

building_config = {
    "adj_matrix": [[0, 1, 1], [1, 0, 0], [1, 0, 0]],
    "expo_ext": [0.0, 1.0, 1.0],
    "t_ext_offset": [0.0, -2.0, 3.0], 
    "start_temp": 19.0,
    "R_val": 0.2, "C_val": 1e6, "R_inter": 0.5, "max_power": 2000, "dt": 60,
    "max_steps": 2880 # 48h
}

if __name__ == "__main__":
    # 1. ENVIRONNEMENT
    raw_env = BuildingEnv(building_config, render_mode=None)
    
    # On utilise 4 environnements virtuels pour stabiliser l'apprentissage (moyenne des gradients)
    env_train = ss.pettingzoo_env_to_vec_env_v1(raw_env)
    # num_cpus=0 = Séquentiel (Rapide), num_vec_envs=4 = Stabilité mathématique
    env_train = ss.concat_vec_envs_v1(env_train, num_vec_envs=4, num_cpus=0, base_class="stable_baselines3")
    env_train = VecMonitor(env_train)

    # 2. MODÈLE PPO
    model = PPO(
        "MlpPolicy",
        env_train,
        verbose=1,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        ent_coef=0.01, # Petite pénalité d'entropie pour encourager l'exploration au début mais pas à la fin
        device="cpu"
    )

    # 3. GO !
    print(f"--- Entraînement PPO (Target: {config['total_timesteps']} steps) ---")
    model.learn(total_timesteps=config["total_timesteps"], progress_bar=True)
    
    os.makedirs("models", exist_ok=True)
    save_path = f"models/{config['model_name']}"
    model.save(save_path)
    print(f"--- Modèle SAUVÉ : {save_path}.zip ---")