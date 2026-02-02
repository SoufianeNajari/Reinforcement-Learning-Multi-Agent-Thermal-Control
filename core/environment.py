import functools
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from .building_model import ThermalModel


class BuildingEnv(ParallelEnv):
    # metadata indispensable pour Stable-Baselines3 et SuperSuit
    metadata = {
        "name": "building_thermal_v0",
        "render_modes": [None]
    }

    def __init__(self, building_config, render_mode=None):
        # 1. Gestion du rendu (Standard PettingZoo)
        self.render_mode = render_mode
        
        # FIX: Filter config to remove keys that ThermalModel doesn't need
        model_config = building_config.copy()
        model_config.pop("max_steps", None) # Remove max_steps before unpacking

        # 2. Initialisation du modèle physique avec unpacking
        self.model = ThermalModel(**model_config)
        
        # 3. Configuration des agents
        self.possible_agents = [f"zone_{i}" for i in range(self.model.nb_zones)]
        
        # 4. Paramètres de simulation
        self.target_temp = 21.0
        self.current_step = 0
        self.default_t_ext = 5.0
        self.alpha = 0.5
        self.beta = 0.01
        self.max_steps = building_config.get("max_steps", 1440)


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
        return observations


    def reset(self, seed=None, options=None):
        # Réinitialisation PettingZoo
        self.agents = self.possible_agents[:]
        self.current_step = 0
        
        # Reset de la physique
        temps = self.model.reset()
        
        # Gestion de t_ext au démarrage
        t_ext = options.get("t_ext", self.default_t_ext) if options else self.default_t_ext
        return self._get_obs(temps, t_ext), {}

    def step(self, actions, t_ext=None):
        # Sécurité si t_ext n'est pas fourni par l'IA (Training)
        if t_ext is None:
            t_ext = self.default_t_ext
            
        price = self.get_price()
        self.current_step += 1
        
        # 1. Calcul de la physique
        act_array = np.array([actions[agent][0] for agent in self.agents])
        new_temps = self.model.step(act_array, t_ext)
        
        # 2. Préparation des observations
        observations = self._get_obs(new_temps, t_ext)
        
        # 3. Calcul de la récompense (Reward)
        rewards = {}
        for i, agent in enumerate(self.agents):
            error = abs(new_temps[i] - self.target_temp)
            thermal_loss = self.alpha * (error + error**2) # Elastic net
            
            power_consumption = abs(act_array[i]) * self.model.max_power # Le froid conssomme autant que le chaud 
            energy_cost = self.beta * (power_consumption * price)
            
            rewards[agent] = -float(thermal_loss + energy_cost) / 10.0

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