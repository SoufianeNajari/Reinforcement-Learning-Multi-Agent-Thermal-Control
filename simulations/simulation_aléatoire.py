import numpy as np
from core.environment import BuildingEnv

# On crée un bâtiment à 3 zones (donc 3 agents)
env = BuildingEnv(nb_zones=3)
observations, infos = env.reset()
time_steps = 20





print("--- Lancement de la simulation de test ---")

# Affichage de l'état initial
for agent in env.agents:
        temp_actuelle = observations[agent][0]
        print(f"  {agent} : {temp_actuelle:.2f}°C")
        

print("--- Test de la Simulation Aléatoire ---")

# On simule time_steps minutes
for step in range(time_steps):
    # Les agents prennent des actions aléatoires (pour tester les limites)
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    
    # Exécution du pas de simulation
    observations, rewards, terminations, truncations, infos = env.step(actions)
    
    print(f"Minute {step}")
    for agent in env.agents:
        temp_actuelle = observations[agent][0]
        print(f"  {agent} : {temp_actuelle:.2f}°C | Reward : {rewards[agent]:.2f}")