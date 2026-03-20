"""
network.py  —  Q-Network (the neural network inside the DQN agent)
------------------------------------------------------------------
Architecture:
    Input layer  : 32 neurons  (16 queue lengths + 16 waiting times, normalised)
    Hidden layer : 128 neurons + ReLU activation
    Hidden layer : 128 neurons + ReLU activation
    Output layer :   4 neurons  (one Q-value per signal phase)

What is a Q-value?
    Q(state, action) = expected total future reward if we take
    'action' in 'state' and then act optimally from there.
    The agent always picks the action with the highest Q-value.

Why two hidden layers?
    Traffic patterns are non-linear. Two layers give enough
    capacity to learn relationships like:
    "West lane is backed up AND North is light → switch to EW green"
"""

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Fully-connected Q-network.

    Parameters
    ----------
    state_size  : int  – length of the observation vector (default 32)
    action_size : int  – number of discrete actions / phases (default 4)
    hidden_size : int  – neurons per hidden layer (default 128)
    """

    def __init__(self, state_size: int = 32, action_size: int = 4, hidden_size: int = 256):
        super(QNetwork, self).__init__()

        self.net = nn.Sequential(
            # Layer 1: state → hidden
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),

            # Layer 2: hidden → hidden
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),

            # Layer 3: hidden → hidden (deeper for better pattern recognition)
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),

            # Output: hidden → Q-values (one per action)
            nn.Linear(hidden_size // 2, action_size),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        state : Tensor of shape (batch_size, state_size)

        Returns
        -------
        Tensor of shape (batch_size, action_size)
            Q-value for every action given the input state.
        """
        return self.net(state)
