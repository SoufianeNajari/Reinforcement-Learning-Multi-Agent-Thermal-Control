# 1. PARAMÈTRES SIMULATION
TARGET_TEMP = 20.0
DT = 60           # Pas de temps (secondes)
MAX_STEPS = 2880  # Durée épisode (mn)
TRAIN_T_EXT = 3.0 # Température extérieure par défaut

# 2. PHYSIQUE DU BÂTIMENT (Partagé Train/Test)
BUILDING_CONFIG = {
    "adj_matrix": [[0, 1, 1], [1, 0, 0], [1, 0, 0]],
    "expo_ext": [0.0, 1.0, 1.0],
    "t_ext_offset": [0.0, -2.0, 3.0], 
    ####
    "start_temp": 15.0, # Température initiale des zones
    
    # Isolation & Inertie
    "R_val": 0.05,       
    "C_int": 5e5,        
    "C_ext": 3e6,     
    "R_inter": 0.1,      
    
    "max_power": 2000,
    "dt": DT,
    "max_steps": MAX_STEPS
}

# 3. RÉGLAGES DE LA RÉCOMPENSE
REWARD_CONFIG = {
    "alpha": 10.0,       # Priorité absolue au confort
    "beta": 1.0          # Coût de l'énergie
}