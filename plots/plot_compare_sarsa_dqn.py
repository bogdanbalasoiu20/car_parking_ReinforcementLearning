# plots/plot_compare_sarsa_dqn.py
import os
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "results"

def load_group(prefixes):
    runs = []
    for filename in os.listdir(RESULTS_DIR):
        if any(filename.startswith(p) for p in prefixes):
            path = os.path.join(RESULTS_DIR, filename)
            runs.append(np.loadtxt(path, delimiter=","))
    return runs

def moving_average(x, window=20):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window)/window, mode="valid")

sarsa_runs = load_group(["sarsa_exp0", "sarsa_exp1", "sarsa_exp2"])
dqn_runs = load_group(["dqn_exp0", "dqn_exp1", "dqn_exp2"])

plt.figure(figsize=(10, 6))

if sarsa_runs:
    min_len = min(len(r) for r in sarsa_runs)
    sarsa_runs = [r[:min_len] for r in sarsa_runs]
    sarsa_mean = np.mean(sarsa_runs, axis=0)
    plt.plot(moving_average(sarsa_mean), label="SARSA (medie pe exp+seed)")

if dqn_runs:
    min_len = min(len(r) for r in dqn_runs)
    dqn_runs = [r[:min_len] for r in dqn_runs]
    dqn_mean = np.mean(dqn_runs, axis=0)
    plt.plot(moving_average(dqn_mean), label="DQN (medie pe exp+seed)")

plt.xlabel("Episod")
plt.ylabel("Reward total")
plt.title("Comparatie SARSA vs DQN (reward mediu)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
