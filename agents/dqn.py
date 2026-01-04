# agents/dqn.py
"""
DQN agent pentru environment-ul parking-v0 (Gymnasium / highway-env)

Acest fisier:
- implementeaza Deep Q-Network (DQN) pentru actiuni discrete
- include Replay Buffer
- include Target Network (stabilizare)
- include selectie epsilon-greedy
- include pas de antrenare (Huber loss / SmoothL1Loss)

Nota: actiunile discrete sunt gestionate in run_dqn.py (ACTIONS).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64

    buffer_size: int = 50_000
    min_buffer_size: int = 2_000

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995  # per episode

    target_update_steps: int = 2_000  # hard update la N pasi de env
    train_every_steps: int = 1        # update la fiecare pas (dupa ce buffer e suficient)

    hidden_dim: int = 128
    seed: int = 42
    device: str = "cpu"


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, seed: int = 0):
        self.capacity = capacity
        self.state_dim = state_dim
        self.rng = np.random.default_rng(seed)

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        idx = self.ptr
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = 1.0 if done else 0.0

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idxs = self.rng.integers(0, self.size, size=batch_size)
        return (
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_states[idxs],
            self.dones[idxs],
        )

    def __len__(self) -> int:
        return self.size


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    def __init__(self, state_dim: int, n_actions: int, cfg: DQNConfig):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.cfg = cfg

        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        self.device = torch.device(cfg.device)

        self.q_online = QNetwork(state_dim, n_actions, cfg.hidden_dim).to(self.device)
        self.q_target = QNetwork(state_dim, n_actions, cfg.hidden_dim).to(self.device)
        self.q_target.load_state_dict(self.q_online.state_dict())
        self.q_target.eval()

        self.optimizer = optim.Adam(self.q_online.parameters(), lr=cfg.lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber

        self.buffer = ReplayBuffer(cfg.buffer_size, state_dim, seed=cfg.seed)

        self.epsilon = cfg.epsilon_start
        self.total_env_steps = 0

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)

        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.q_online(s)[0]
            return int(torch.argmax(q).item())

    def store_transition(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.buffer.push(state, action, reward, next_state, done)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.cfg.epsilon_end, self.epsilon * self.cfg.epsilon_decay)

    def maybe_update_target(self) -> None:
        if self.total_env_steps % self.cfg.target_update_steps == 0:
            self.q_target.load_state_dict(self.q_online.state_dict())

    def train_step(self) -> float | None:
        if len(self.buffer) < self.cfg.min_buffer_size:
            return None

        if self.total_env_steps % self.cfg.train_every_steps != 0:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.cfg.batch_size)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_sa = self.q_online(states_t).gather(1, actions_t)

        with torch.no_grad():
            q_next = self.q_target(next_states_t).max(dim=1, keepdim=True).values
            target = rewards_t + self.cfg.gamma * (1.0 - dones_t) * q_next

        loss = self.loss_fn(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_online.parameters(), max_norm=10.0)
        self.optimizer.step()

        return float(loss.item())
