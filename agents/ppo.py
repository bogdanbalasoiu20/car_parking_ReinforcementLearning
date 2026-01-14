import os
import sys
import csv
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, StopTrainingOnMaxEpisodes
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from environment.environment import create_environment

# Configurare experiment
EXP_NAME = "ppo_parking"
SEED = 0
EPISODES = 500

def run_ppo_experiment(learning_rate, n_steps, gamma, seed, episodes=500):
    
    set_random_seed(seed)

    def _make_env():
        env = create_environment(render=False, seed=seed)
        env = TimeLimit(env, max_episode_steps=1500)
        return env

    env = DummyVecEnv([_make_env])
    env = VecMonitor(env)

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        gamma=gamma,
        seed=seed,
        verbose=0 
    )

    # 1. Callback pentru colectarea datelor
    class DataCollectionCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.rewards = []
            self.timesteps = []

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            for info in infos:
                if "episode" in info:
                    self.rewards.append(float(info["episode"]["r"]))
                    self.timesteps.append(self.num_timesteps)
            return True

    data_callback = DataCollectionCallback()

    # 2. Callback NOU: Opreste antrenarea exact cand atingem nr de episoade
    stop_callback = StopTrainingOnMaxEpisodes(max_episodes=episodes, verbose=1)

    callbacks = CallbackList([data_callback, stop_callback])

    print(f"--- Start PPO (lr={learning_rate}, Seed={seed}) ---")
    
    # Setam un numar urias de pasi (ex: 1 milion) dar 'stop_callback' 
    # va opri totul cand ajungem la 500 de episoade.
    model.learn(total_timesteps=1_000_000, callback=callbacks, progress_bar=False)

    env.close()
    return data_callback.rewards, data_callback.timesteps

def run_all_experiments():
    experiments = [
        {"lr": 3e-4, "n_steps": 1024, "gamma": 0.99}, 
        {"lr": 1e-4, "n_steps": 1024, "gamma": 0.99},
        {"lr": 3e-4, "n_steps": 2048, "gamma": 0.99},
    ]

    seeds = [0, 42, 123]
    output_dir = "results_ppo"
    os.makedirs(output_dir, exist_ok=True)

    for exp_id, params in enumerate(experiments):
        for seed in seeds:
            rewards, steps = run_ppo_experiment(
                learning_rate=params["lr"],
                n_steps=params["n_steps"],
                gamma=params["gamma"],
                seed=seed,
                episodes=500
            )

            filename = f"{output_dir}/ppo_exp{exp_id}_seed{seed}.csv"
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Episode", "Total_Timesteps", "Reward"])
                for i, (r, t) in enumerate(zip(rewards, steps)):
                    writer.writerow([i + 1, t, r])

            print(f"[OK] Salvat: {filename} ({len(rewards)} episoade)")

if __name__ == "__main__":
    run_all_experiments()