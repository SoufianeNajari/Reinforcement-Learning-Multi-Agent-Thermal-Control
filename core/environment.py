import functools
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from .building_model import ThermalModel
import config as cfg


class BuildingEnv(ParallelEnv):
    # metadata indispensable pour Stable-Baselines3 et SuperSuit
    metadata = {
        "name": "building_thermal_v0",
        "render_modes": [None]
    }

    def __init__(self, building_config, render_mode=None, random_start=True):

        if building_config is None:
            building_config = cfg.BUILDING_CONFIG

        # 1. Gestion du rendu (Standard PettingZoo)
        self.render_mode = render_mode
        
        model_config = building_config.copy()
        model_config.pop("max_steps", None) # Remove max_steps before unpacking

        # 2. Initialisation du modèle physique avec unpacking
        self.random_start = random_start
        self.model = ThermalModel(**model_config)
        
        # 3. Configuration des agents
        self.possible_agents = [f"zone_{i}" for i in range(self.model.nb_zones)]
        
        # 4. Paramètres de simulation
        self.target_temp = cfg.TARGET_TEMP
        self.current_step = 0
        self.default_t_ext = cfg.TRAIN_T_EXT
        
        self.alpha = cfg.REWARD_CONFIG["alpha"]
        self.beta = cfg.REWARD_CONFIG["beta"]
        self.max_steps = building_config.get("max_steps", cfg.MAX_STEPS)


    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # [Température zone, Température extérieure]
        return spaces.Box(low=-20, high=120, shape=(5,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        # Action continue entre -1 (froid) et 1 (chaud)
        return spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

    def get_price(self):
        minute_of_day = self.current_step % 1440
        if minute_of_day < 360 or minute_of_day > 1320:
            return 0.15
        return 0.25
    """
    def _get_obs(self, temps, base_t_ext):
        price = self.get_price()
        observations = {}
        for i, agent in enumerate(self.possible_agents):
            idx_voisins = np.where(self.model.adj[i] == 1)[0]
            nb_v = len(idx_voisins)
            sum_v_temp = np.sum(temps[idx_voisins]) if nb_v > 0 else 0.0
            t_ext_zone = base_t_ext + self.model.t_ext_offset[i]
            
            observations[agent] = np.array([
                temps[i],
                t_ext_zone,
                sum_v_temp,
                float(nb_v),
                price
            ], dtype=np.float32)
        return observations """

    def _get_obs(self, temps, base_t_ext):
        price = self.get_price()
        observations = {}
        for i, agent in enumerate(self.possible_agents):
            # Données locales
            my_temp = temps[i]
            t_ext_zone = base_t_ext + self.model.t_ext_offset[i]
            
            # Données voisins
            idx_voisins = np.where(self.model.adj[i] == 1)[0]
            if len(idx_voisins) > 0:
                mean_v_temp = np.mean(temps[idx_voisins])
            else:
                mean_v_temp = my_temp # Si pas de voisin, on prend sa propre temp

            # --- ON CENTRE TOUT SUR 0 ---
            observations[agent] = np.array([
                my_temp - self.target_temp,        # Ecart Cible (Ex: -1.0 si 20°C)
                t_ext_zone - my_temp,              # Ecart Extérieur (Delta T qui cause les pertes)
                mean_v_temp - my_temp,             # Ecart Voisin (Delta T échange)
                price,                             # Prix (déjà petit, ok)
                float(len(idx_voisins))            # Info structurelle
            ], dtype=np.float32)
            
        return observations

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.current_step = 0
        
        if self.random_start:
            self.model.start_temp = np.random.uniform(10.0, 35.0)
            self.default_t_ext = np.random.uniform(-5.0, 35.0)
        
        temps = self.model.reset()
        
        t_ext = options.get("t_ext", self.default_t_ext) if options else self.default_t_ext
        return self._get_obs(temps, t_ext), {}

    def step(self, actions, t_ext=None):

        if t_ext is None:
            t_ext = self.default_t_ext
            
        price = self.get_price()
        self.current_step += 1
        
        # 1. Calcul de la physique
        act_array = np.array([actions[agent][0] for agent in self.agents])
        new_temps = self.model.step(act_array, t_ext)
        
        # 2. Préparation des observations
        observations = self._get_obs(new_temps, t_ext)
        
        # 3. Calcul du reward
        rewards = {}
        for i, agent in enumerate(self.agents):
            error = abs(new_temps[i] - self.target_temp)
            thermal_loss = self.alpha * (error + (error)**2) # Elastic net
            
            power_consumption = abs(act_array[i]) * self.model.max_power # Le froid conssomme autant que le chaud 
            energy_cost = self.beta * (power_consumption * price) / 60.0 # Coût en €/min
            
            rewards[agent] = -float(thermal_loss + energy_cost) / 20  # Normalisation

        # 4. Conditions d'arrêt (Truncation après 24h)
        # Indispensable pour que SB3 affiche 'ep_rew_mean'
        duree_max_atteinte = self.current_step >= self.max_steps
        
        # Indispensable : On définit l'état pour TOUS les agents possibles
        terminations = {agent: False for agent in self.possible_agents}
        truncations = {agent: duree_max_atteinte for agent in self.possible_agents}
        infos = {agent: {} for agent in self.possible_agents}
        
        # Logique PettingZoo : si c'est fini, on vide la liste self.agents
        if duree_max_atteinte:
            self.agents = []

        return observations, rewards, terminations, truncations, infos