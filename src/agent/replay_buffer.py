"""
replay_buffer.py  —  Experience Replay Memory
----------------------------------------------
Stores past (state, action, reward, next_state, done) transitions
and provides random mini-batch sampling for DQN training.

Why experience replay?
    Neural networks trained on sequential data develop correlations
    between consecutive samples, which destabilises learning.
    By storing experiences in a buffer and sampling randomly, we:
        1. Break time correlations between samples.
        2. Re-use past experiences multiple times (data efficiency).
        3. Stabilise the loss gradient across training steps.

Buffer behaviour:
    - Fixed maximum size (capacity). When full, oldest entries are
      overwritten (circular / ring buffer via deque with maxlen).
    - Sampling only starts once the buffer has at least `batch_size`
      entries, so the agent explores before it starts learning.
"""

import random
import numpy as np
from collections import deque


class ReplayBuffer:
    """
    Fixed-size circular buffer for experience replay.

    Parameters
    ----------
    capacity   : int  – maximum number of transitions to store
    batch_size : int  – number of transitions returned per sample call
    """

    def __init__(self, capacity: int = 10_000, batch_size: int = 64):
        self.buffer     = deque(maxlen=capacity)   # auto-drops oldest when full
        self.batch_size = batch_size

    # ── Public API ───────────────────────────────────────────────────────────

    def push(self, state, action, reward, next_state, done):
        """
        Save one transition to the buffer.

        Parameters
        ----------
        state      : np.ndarray  – observation before action
        action     : int         – index of the chosen phase (0-3)
        reward     : float       – reward received after action
        next_state : np.ndarray  – observation after action
        done       : bool        – True if the episode ended
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self):
        """
        Randomly sample a mini-batch of transitions.

        Returns
        -------
        Tuple of five np.ndarrays:
            states, actions, rewards, next_states, dones
        Each array has shape (batch_size, ...).
        Returns None if the buffer doesn't have enough samples yet.
        """
        if not self.ready():
            return None

        batch = random.sample(self.buffer, self.batch_size)

        # Unzip the list of tuples into separate arrays
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),   # 1.0 = done, 0.0 = not done
        )

    def ready(self) -> bool:
        """Return True when enough experiences exist to start training."""
        return len(self.buffer) >= self.batch_size

    def __len__(self) -> int:
        return len(self.buffer)
