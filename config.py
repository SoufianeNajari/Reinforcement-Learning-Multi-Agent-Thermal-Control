# 1. PARAMÈTRES SIMULATION
TARGET_TEMP = 20.0
DT = 60           # Pas de temps (secondes)
MAX_STEPS = 2880  # Durée épisode (mn)
#TRAIN_T_EXT = -5.0 # Scénario hivernal
TRAIN_T_EXT = 35.0 # Scénario estival

# 2. PHYSIQUE DU BÂTIMENT
BUILDING_CONFIG = {
    "adj_matrix": [[0, 1, 1], [1, 0, 0], [1, 0, 0]],
    "expo_ext": [0.0, 1.0, 1.0],
    "t_ext_offset": [0.0, -2.0, 3.0], 
    ####
    #"start_temp": 10.0, # Scénario hivernal
    "start_temp": 35.0, # Scénario estival

    # Isolation & Inertie
    "R_val": 0.05, # Résistance thermique avec l'extérieur
    "C_int": 1e5,        
    "C_ext": 5e5,     
    "R_inter": 0.1, # Résistance thermique entre les zones   
    
    "max_power": 2000,
    "dt": DT,
    "max_steps": MAX_STEPS
}

# 3. REWARDS
REWARD_CONFIG = {
    "alpha": 2.0,       # Importance au confort
    "beta": 1.0          # Importance de l'économie d'énergie
}