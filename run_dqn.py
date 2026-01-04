# run_dqn.py
"""
Runner pentru DQN pe parking-v0:
- defineste actiunile discrete (aceleasi ca la SARSA)
- ruleaza experimente multiple si seeds
- salveaza reward-urile in results/*.csv

Ruleaza:
python run_dqn.py
"""

import os
import csv
import random
import numpy as np

import torch

from environment import create_environment
from agents.dqn import DQNAgent, DQNConfig


ACTIONS = [
    np.array([ 1.0,  0.0], dtype=np.float32),   # inainte
    np.array([-1.0,  0.0], dtype=np.float32),   # inapoi
    np.array([ 0.0,  1.0], dtype=np.float32),   # stanga
    np.array([ 0.0, -1.0], dtype=np.float32),   # dreapta
    np.array([ 0.0,  0.0], dtype=np.float32)    # stop
]
N_ACTIONS = len(ACTIONS)


def extract_state(obs) -> np.ndarray:
    """
    obs este dict returnat de parking-v0.
    Vectorul de stare este in obs["observation"].
    """
    return np.array(obs["observation"], dtype=np.float32)


def run_dqn_experiment(
    cfg: DQNConfig,
    seed: int,
    episodes: int = 500,
    max_steps: int = 300,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = create_environment(render=False, seed=seed)

    obs, _ = env.reset()
    state = extract_state(obs)
    state_dim = int(state.shape[0])

    agent_cfg = cfg
    agent_cfg.seed = seed
    agent = DQNAgent(state_dim=state_dim, n_actions=N_ACTIONS, cfg=agent_cfg)

    episode_rewards = []
    episode_losses = []

    for ep in range(episodes):
        obs, _ = env.reset()
        state = extract_state(obs)

        total_reward = 0.0
        losses = []

        for _ in range(max_steps):
            action_idx = agent.select_action(state)
            obs2, reward, terminated, truncated, _info = env.step(ACTIONS[action_idx])
            done = bool(terminated or truncated)

            next_state = extract_state(obs2)

            agent.store_transition(state, action_idx, float(reward), next_state, done)

            agent.total_env_steps += 1
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

            agent.maybe_update_target()

            state = next_state
            total_reward += float(reward)

            if done:
                break

        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        episode_losses.append(float(np.mean(losses)) if losses else np.nan)

    env.close()
    return episode_rewards, episode_losses


def run_all_experiments():
    os.makedirs("results", exist_ok=True)

    experiments = [
        {"lr": 1e-3,  "batch": 64,  "target_update": 2000, "eps_decay": 0.995},
        {"lr": 5e-4,  "batch": 64,  "target_update": 2000, "eps_decay": 0.995},
        {"lr": 1e-3,  "batch": 128, "target_update": 1000, "eps_decay": 0.995},
    ]

    seeds = [0, 42, 123]

    for exp_id, p in enumerate(experiments):
        for seed in seeds:
            cfg = DQNConfig(
                lr=p["lr"],
                batch_size=p["batch"],
                target_update_steps=p["target_update"],
                epsilon_decay=p["eps_decay"],
                device="cpu",
            )

            rewards, losses = run_dqn_experiment(cfg=cfg, seed=seed)

            rewards_file = f"results/dqn_exp{exp_id}_seed{seed}.csv"
            with open(rewards_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(rewards)

            losses_file = f"results/dqn_loss_exp{exp_id}_seed{seed}.csv"
            with open(losses_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(losses)

            print(f"[OK] DQN exp{exp_id}, seed {seed} -> {rewards_file} (+loss in {losses_file})")


if __name__ == "__main__":
    run_all_experiments()
