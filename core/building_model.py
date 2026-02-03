import numpy as np
T_EXT = 5.0  # Température extérieure constante pour la simulation


class ThermalModel:
    def __init__(self, adj_matrix, expo_ext, t_ext_offset, start_temp, R_val, C_int, C_ext, R_inter, max_power, dt):
        
        self.adj = np.array(adj_matrix)
        self.nb_zones = self.adj.shape[0]
        self.expo_ext = np.array(expo_ext)
        self.t_ext_offset = np.array(t_ext_offset)
        self.start_temp = start_temp
        self.R = R_val
        self.R_inter = R_inter
        self.max_power = max_power
        self.dt = dt
        self.temp_interne = np.full(self.nb_zones, self.start_temp)

        # Capacité thermique totale par zone
        self.C = np.full(self.nb_zones, C_int, dtype=np.float32)
        self.C += self.expo_ext * C_ext

    def step(self, actions, base_t_ext):

        # 1. Flux avec l'extérieur (Loi de Fourier)
        z_t_ext = base_t_ext + self.t_ext_offset # Temp extérieure pour chaque zone désigné comme base + offset ( ex : 5 (base) + 2(offset) = 7°C )
        q_hvac = actions * self.max_power
        flux_ext = self.expo_ext * (z_t_ext - self.temp_interne) / self.R
        
        # 2. Flux entre les zones (Inter-zone)
        diff_temp = self.temp_interne[None, :] - self.temp_interne[:, None]
        flux_inter = np.sum(diff_temp * self.adj, axis=1) / self.R_inter
        
        flux_total = flux_ext + q_hvac + flux_inter

        # 3. Application de la méthode d'Euler
        # dT/dt = (Flux_total) / C
        dT = flux_total / self.C * self.dt
        self.temp_interne += dT
        
        return self.temp_interne.copy()

    def reset(self):
        self.temp_interne = np.full(self.nb_zones, self.start_temp)
        return self.temp_interne.copy()