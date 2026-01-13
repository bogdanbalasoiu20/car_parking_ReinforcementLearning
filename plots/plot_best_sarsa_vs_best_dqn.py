import os
import re
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "results"

def moving_average(x, window=20):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window)/window, mode="valid")

def load_runs(pattern):
    """
    pattern: regex cu grupuri exp_id si seed
    return: dict exp_id -> list[runs]
    """
    runs = {}
    for fn in os.listdir(RESULTS_DIR):
        m = re.match(pattern, fn)
        if not m:
            continue
        exp_id = int(m.group(1))
        path = os.path.join(RESULTS_DIR, fn)
        r = np.loadtxt(path, delimiter=",")
        runs.setdefault(exp_id, []).append(r)
    return runs

def mean_curve(runs):
    min_len = min(len(r) for r in runs)
    runs = [r[:min_len] for r in runs]
    return np.mean(runs, axis=0)

def score_curve(curve, last_k=50):
    # scor: media ultimelor last_k episoade (mai mare = mai bun)
    if len(curve) < last_k:
        return float(np.mean(curve))
    return float(np.mean(curve[-last_k:]))

# incarcam pe experimente
sarsa_by_exp = load_runs(r"sarsa_exp(\d+)_seed\d+\.csv$")
dqn_by_exp   = load_runs(r"dqn_exp(\d+)_seed\d+\.csv$")

# alegem "best exp" dupa media ultimelor 50 episoade
best_sarsa_exp = max(sarsa_by_exp.keys(), key=lambda e: score_curve(mean_curve(sarsa_by_exp[e])))
best_dqn_exp   = max(dqn_by_exp.keys(),   key=lambda e: score_curve(mean_curve(dqn_by_exp[e])))

sarsa_mean = mean_curve(sarsa_by_exp[best_sarsa_exp])
dqn_mean   = mean_curve(dqn_by_exp[best_dqn_exp])

plt.figure(figsize=(10, 6))
plt.plot(moving_average(sarsa_mean), label=f"SARSA best exp={best_sarsa_exp}")
plt.plot(moving_average(dqn_mean),   label=f"DQN best exp={best_dqn_exp}")

plt.xlabel("Episod")
plt.ylabel("Reward total")
plt.title("Comparatie BEST SARSA vs BEST DQN (medie pe seed-uri)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

print(f"[INFO] Best SARSA exp={best_sarsa_exp}, score(last50)={score_curve(sarsa_mean):.3f}")
print(f"[INFO] Best DQN   exp={best_dqn_exp}, score(last50)={score_curve(dqn_mean):.3f}")
