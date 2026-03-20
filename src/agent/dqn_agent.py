"""
dqn_agent.py  —  Deep Q-Network Agent
--------------------------------------
Implements the full DQN algorithm (Mnih et al., 2015) with:
    • Epsilon-greedy exploration
    • Experience replay  (via ReplayBuffer)
    • Separate policy and target networks
    • Periodic target-network updates (hard copy)
    • Model save / load

Learning cycle (called every decision step):
    1. Observe state  s
    2. Choose action  a  (epsilon-greedy)
    3. Execute action → get reward r, next state s'
    4. Store (s, a, r, s', done) in replay buffer
    5. Sample random mini-batch from buffer
    6. Compute Bellman target:
           y = r                          if done
           y = r + γ · max Q_target(s')  otherwise
    7. Update policy network:  minimise (y − Q_policy(s,a))²
    8. Every C steps: copy policy weights → target network
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .network       import QNetwork
from .replay_buffer import ReplayBuffer


class DQNAgent:
    """
    DQN Agent for adaptive traffic signal control.

    Parameters
    ----------
    state_size       : int   – observation vector length (32)
    action_size      : int   – number of signal phases    (4)
    lr               : float – Adam learning rate
    gamma            : float – discount factor (how much future rewards matter)
    epsilon_start    : float – initial exploration rate  (100% random)
    epsilon_end      : float – minimum exploration rate  (5% random)
    epsilon_decay    : float – multiplicative decay per episode
    buffer_capacity  : int   – replay buffer size
    batch_size       : int   – training mini-batch size
    target_update    : int   – copy policy→target every N training steps
    device           : str   – 'cpu' or 'cuda'
    """

    def __init__(
        self,
        state_size:      int   = 32,
        action_size:     int   = 4,
        hidden_size:     int   = 256,
        lr:              float = 5e-4,
        gamma:           float = 0.99,
        epsilon_start:   float = 1.0,
        epsilon_end:     float = 0.05,
        epsilon_decay:   float = 0.995,
        buffer_capacity: int   = 10_000,
        batch_size:      int   = 64,
        target_update:   int   = 10,
        device:          str   = "cpu",
    ):
        self.state_size  = state_size
        self.action_size = action_size
        self.gamma       = gamma
        self.epsilon     = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size  = batch_size
        self.target_update = target_update
        self.device      = torch.device(device)

        # ── Two networks ─────────────────────────────────────────────────────
        # policy_net : trained every step  (fast-moving)
        # target_net : updated every C steps (stable reference for Bellman target)
        self.policy_net = QNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_net = QNetwork(state_size, action_size, hidden_size).to(self.device)
        self._sync_target()                    # start with identical weights
        self.target_net.eval()                 # target never back-propagates

        # ── Optimiser & loss ─────────────────────────────────────────────────
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn   = nn.SmoothL1Loss()   # Huber loss — stable when Q-values are large

        # ── Replay buffer ────────────────────────────────────────────────────
        self.memory = ReplayBuffer(capacity=buffer_capacity, batch_size=batch_size)

        # ── Counters ─────────────────────────────────────────────────────────
        self._train_steps = 0   # total gradient updates (for target sync)

    # ── Action selection ─────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray) -> int:
        """
        Epsilon-greedy policy.

        With probability epsilon  → pick a random phase  (explore)
        With probability 1-epsilon → pick argmax Q(s,·)  (exploit)

        Epsilon starts at 1.0 (pure random) and decays each episode
        until it reaches epsilon_end (mostly greedy).
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)   # random action

        # Convert state to tensor and get Q-values from policy network
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)     # shape: (1, action_size)
        return q_values.argmax(dim=1).item()             # index of highest Q

    # ── Memory ───────────────────────────────────────────────────────────────

    def store(self, state, action, reward, next_state, done):
        """Push one transition into the replay buffer."""
        self.memory.push(state, action, reward, next_state, done)

    # ── Training step ────────────────────────────────────────────────────────

    def train(self) -> float | None:
        """
        Sample a mini-batch and perform one gradient update.

        Returns the scalar loss value, or None if the buffer isn't
        ready yet (not enough samples collected).

        Bellman update:
            y_i = r_i                              if done_i
            y_i = r_i + γ · max_a Q_target(s'_i)  otherwise

            Loss = MSE( y_i,  Q_policy(s_i, a_i) )
        """
        batch = self.memory.sample()
        if batch is None:
            return None     # not enough experience yet

        states, actions, rewards, next_states, dones = batch

        # Move everything to the target device
        states      = torch.FloatTensor(states)     .to(self.device)
        actions     = torch.LongTensor(actions)     .to(self.device)
        rewards     = torch.FloatTensor(rewards)    .to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones       = torch.FloatTensor(dones)      .to(self.device)

        # ── Current Q-values: Q_policy(s, a) ─────────────────────────────
        # policy_net outputs Q for all actions; we select the one taken
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # ── Bellman target ────────────────────────────────────────────────
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(dim=1)[0]
            target_q   = rewards + self.gamma * max_next_q * (1.0 - dones)

        # ── Gradient update ───────────────────────────────────────────────
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # ── Sync target network every `target_update` steps ──────────────
        self._train_steps += 1
        if self._train_steps % self.target_update == 0:
            self._sync_target()

        return loss.item()

    # ── Epsilon decay ────────────────────────────────────────────────────────

    def decay_epsilon(self):
        """
        Call once per episode after training.
        Reduces epsilon so the agent gradually shifts from
        exploration (random) to exploitation (learned policy).
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # ── Save / Load ──────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save the policy network weights and agent state to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state":   self.optimizer.state_dict(),
            "epsilon":           self.epsilon,
            "train_steps":       self._train_steps,
        }, path)
        print(f"[Agent] Model saved → {path}")

    def load(self, path: str):
        """Load weights from a previously saved checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.epsilon      = checkpoint["epsilon"]
        self._train_steps = checkpoint["train_steps"]
        print(f"[Agent] Model loaded ← {path}")

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _sync_target(self):
        """Hard copy: policy network weights → target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
