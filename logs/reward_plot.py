import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

log_dir = "logs/"
model_name = "PPO_v5_HVAC"
monitor_file = os.path.join(log_dir, f"{model_name}.monitor.csv")

data = pd.read_csv(monitor_file, skiprows=1)

burn_in_episodes = max(1, int(len(data) * 0.05))
data = data.iloc[burn_in_episodes:]

x = np.cumsum(data['l'].values)
y = data['r'].values

plt.figure(figsize=(10, 6))
plt.plot(x, y, alpha=0.3, color='blue', label='Raw Reward')

if len(y) >= 10:
    window_size = 10
    y_smooth = np.convolve(y, np.ones(window_size)/window_size, mode='valid')
    x_smooth = x[window_size-1:]
    plt.plot(x_smooth, y_smooth, color='blue', linewidth=2, label='Smoothed Reward')

plt.yscale('symlog') 
plt.xlabel('Timesteps')
plt.ylabel('Cumulative Reward (Symlog Scale)')
plt.title(f'Learning Curve (Burn-in + Symlog) - {model_name}')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)

save_path = os.path.join(log_dir, "learning_curve_clean.png")
plt.savefig(save_path)