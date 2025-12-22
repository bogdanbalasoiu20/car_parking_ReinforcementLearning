import os
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "results"

# structura: alpha -> lista de rulari (fiecare rulare = lista de reward-uri)
experiments = {
    "alpha=0.05": [],
    "alpha=0.1": [],
    "alpha=0.2": []
}

# mapare manuala
alpha_mapping = {
    "sarsa_exp0": "alpha=0.05",
    "sarsa_exp1": "alpha=0.1",
    "sarsa_exp2": "alpha=0.2",
}

# citim toate fisierele CSV
for filename in os.listdir(RESULTS_DIR):
    for exp_name, alpha_name in alpha_mapping.items():
        if filename.startswith(exp_name):
            path = os.path.join(RESULTS_DIR, filename)
            rewards = np.loadtxt(path, delimiter=",")
            experiments[alpha_name].append(rewards)

# functie de smoothing (media glisanta)
def moving_average(x, window=20):
    return np.convolve(x, np.ones(window)/window, mode="valid")

# plot
plt.figure(figsize=(10, 6))

for alpha, runs in experiments.items():
    if not runs:
        continue

    min_len = min(len(r) for r in runs)
    runs = [r[:min_len] for r in runs]

    mean_rewards = np.mean(runs, axis=0)
    smoothed = moving_average(mean_rewards)

    plt.plot(smoothed, label=alpha)

plt.xlabel("Episod")
plt.ylabel("Reward total")
plt.title("SARSA – evolutia reward-ului (medie pe seed-uri)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
