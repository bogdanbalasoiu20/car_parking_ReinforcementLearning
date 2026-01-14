import os
import numpy as np
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.normpath(os.path.join(_HERE, "..", "results_ppo"))

exp_mapping = {
    "ppo_exp0": "exp0",
    "ppo_exp1": "exp1",
    "ppo_exp2": "exp2",
}

experiments = {label: [] for label in exp_mapping.values()}

if os.path.isdir(RESULTS_DIR):
    for filename in os.listdir(RESULTS_DIR):
        for exp_name, label in exp_mapping.items():
            if filename.startswith(exp_name):
                path = os.path.join(RESULTS_DIR, filename)
                
                try:
                    rewards = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(2,))
                    
                    if rewards.ndim == 0:
                        rewards = np.array([rewards])
                        
                    experiments[label].append(rewards)
                    print(f"[Incarcat] {filename} -> {label}")
                except Exception as e:
                    print(f"[Eroare] Nu am putut incarca {filename}: {e}")
else:
    print(f"[Eroare] Nu am gasit folderul de rezultate: {RESULTS_DIR}")

def moving_average(x, window=20):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")

plt.figure(figsize=(10, 6))

found_data = False
for label, runs in experiments.items():
    if not runs:
        continue
    
    found_data = True
    min_len = min(len(r) for r in runs)
    if min_len == 0: 
        continue

    runs_truncated = [r[:min_len] for r in runs]
    mean_rewards = np.mean(runs_truncated, axis=0)

    smoothed_mean = moving_average(mean_rewards)

    x_axis = range(len(smoothed_mean))
    line = plt.plot(x_axis, smoothed_mean, label=label)

if found_data:
    plt.xlabel("Episod")
    plt.ylabel("Reward Total")
    plt.title("PPO Parking - Evolutia Antrenarii")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print("Nu s-au gasit date! Ruleaza intai ppo.py")