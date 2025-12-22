"""
SARSA agent pentru environment-ul parking-v0 (Gymnasium / highway-env)

Acest fisier:
- implementeaza SARSA (algoritm tabular)
- ruleaza experimente multiple
- permite reglarea hiperparametrilor
- salveaza rezultate pentru grafice si tabele
"""

import numpy as np
import random
import csv
import os

from environment import create_environment

#Discretizam starea( sarsa nu poate lucra cu stari continue -> transformam observatia in stare discreta(bins))
NUM_BINS = 10


def discretize(value, min_val, max_val, bins=NUM_BINS):
    value = np.clip(value, min_val, max_val)
    return int((value - min_val) / (max_val - min_val) * bins)


def discretize_state(obs):
    """
    obs este un dictionar returnat de parking-v0.
    Vectorul de stare real este in obs["observation"].
    """

    obs_vec = obs["observation"]

    x, y = obs_vec[0], obs_vec[1]

    # orientarea este data de cos(theta) si sin(theta)
    cos_t, sin_t = obs_vec[4], obs_vec[5]
    theta = np.arctan2(sin_t, cos_t)

    return (
        discretize(x, -1, 1),
        discretize(y, -1, 1),
        discretize(theta, -np.pi, np.pi)
    )



#definim un set finit de actiuni pentru environment
ACTIONS = [
    np.array([ 1.0,  0.0]),   # inainte
    np.array([-1.0,  0.0]),   # inapoi
    np.array([ 0.0,  1.0]),   # stanga
    np.array([ 0.0, -1.0]),   # dreapta
    np.array([ 0.0,  0.0])    # stop
]

N_ACTIONS = len(ACTIONS)


# 3. politica ε-greedy
def choose_action(state, Q, epsilon):
    if random.random() < epsilon:
        return random.randrange(N_ACTIONS)
    return max(range(N_ACTIONS), key=lambda a: Q.get((state, a), 0.0))


# rulam un experiment
def run_sarsa_experiment(
    alpha,
    gamma,
    epsilon_start,
    epsilon_decay,
    seed,
    episodes=500,
    max_steps=300
):
    """
    Ruleaza un experiment SARSA cu un set fix de hiperparametri
    si returneaza reward-ul total per episod.
    """

    # setam seed-urile pentru reproductibilitate
    random.seed(seed)
    np.random.seed(seed)

    env = create_environment(render=False, seed=seed)

    Q = {}  # Q-table
    epsilon = epsilon_start
    episode_rewards = []

    for episode in range(episodes):

        obs, _ = env.reset()
        state = discretize_state(obs)
        action = choose_action(state, Q, epsilon)

        total_reward = 0

        for _ in range(max_steps):

            obs, reward, terminated, truncated, _ = env.step(ACTIONS[action])
            done = terminated or truncated

            next_state = discretize_state(obs)
            next_action = choose_action(next_state, Q, epsilon)

            # formula sarsa
            old_q = Q.get((state, action), 0.0)
            next_q = Q.get((next_state, next_action), 0.0)

            Q[(state, action)] = old_q + alpha * (reward + gamma * next_q - old_q)

            state = next_state
            action = next_action
            total_reward += reward

            if done:
                break

        # scadem explorarea
        epsilon = max(0.05, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)

    env.close()
    return episode_rewards



# experimente multiple
def run_all_experiments():
    """
    Ruleaza mai multe experimente SARSA cu hiperparametri diferiti
    pentru analiza stabilitatii si convergentei.
    """

    experiments = [
        {"alpha": 0.05, "gamma": 0.99, "epsilon": 1.0, "decay": 0.995},
        {"alpha": 0.10, "gamma": 0.99, "epsilon": 1.0, "decay": 0.995},
        {"alpha": 0.20, "gamma": 0.99, "epsilon": 1.0, "decay": 0.995},
    ]

    seeds = [0, 42, 123]

    os.makedirs("results", exist_ok=True)

    for exp_id, params in enumerate(experiments):
        for seed in seeds:

            rewards = run_sarsa_experiment(
                alpha=params["alpha"],
                gamma=params["gamma"],
                epsilon_start=params["epsilon"],
                epsilon_decay=params["decay"],
                seed=seed
            )

            filename = f"results/sarsa_exp{exp_id}_seed{seed}.csv"
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(rewards)

            print(f"[OK] Experiment {exp_id}, seed {seed} salvat -> {filename}")


if __name__ == "__main__":
    run_all_experiments()
