# plots/plot_dqn_results.py
import os
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "results"

experiments = {
    "exp0": [],
    "exp1": [],
    "exp2": [],
}

exp_mapping = {
    "dqn_exp0": "exp0",
    "dqn_exp1": "exp1",
    "dqn_exp2": "exp2",
}

for filename in os.listdir(RESULTS_DIR):
    for exp_name, key in exp_mapping.items():
        if filename.startswith(exp_name) and filename.endswith(".csv"):
            path = os.path.join(RESULTS_DIR, filename)
            rewards = np.loadtxt(path, delimiter=",")
            experiments[key].append(rewards)

def moving_average(x, window=20):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window)/window, mode="valid")

plt.figure(figsize=(10, 6))

for exp, runs in experiments.items():
    if not runs:
        continue

    min_len = min(len(r) for r in runs)
    runs = [r[:min_len] for r in runs]

    mean_rewards = np.mean(runs, axis=0)
    smoothed = moving_average(mean_rewards)

    plt.plot(smoothed, label=exp)

plt.xlabel("Episod")
plt.ylabel("Reward total")
plt.title("DQN – evolutia reward-ului (medie pe seed-uri)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
