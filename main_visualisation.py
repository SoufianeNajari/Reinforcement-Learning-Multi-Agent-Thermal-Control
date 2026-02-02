import os
import pandas as pd
import matplotlib.pyplot as plt

FILE_TO_PLOT = "results/data_PPO_v1.csv" #CHANGER ICI SELON LE FICHIER À PLOTER
df = pd.read_csv(FILE_TO_PLOT)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

temp_cols = [c for c in df.columns if c.startswith("temp_")]
for col in temp_cols:
    ax1.plot(df["step"], df[col], label=col.replace("temp_", ""))

ax1.axhline(y=df["target"].iloc[0], color='r', linestyle='--', label="Consigne")
ax1.set_ylabel("Température (°C)")
ax1.set_title("Réponse du système")
ax1.legend()
ax1.grid(True)

act_cols = [c for c in df.columns if c.startswith("act_")]
for col in act_cols:
    ax2.step(df["step"], df[col], label=col.replace("act_", ""))

ax2.set_ylabel("Action PAC")
ax2.set_xlabel("Temps (min)")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

os.makedirs("graphs", exist_ok=True)
plt.savefig("graphs/resultat_simulation.png")
print("Graphique sauvegardé dans graphs/resultat_simulation.png")