"""
metrics.py  —  Episode metrics tracker + plotter
-------------------------------------------------
Tracks per-episode statistics during training and saves:
    • results/logs/training_log.csv   — raw numbers (every episode)
    • results/plots/training.png      — 4-panel learning curve chart
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt


class MetricsTracker:
    """
    Records one row of stats per episode and writes plots on demand.

    Tracked per episode
    -------------------
    episode       : episode number
    total_reward  : sum of rewards across all decision steps
    avg_wait      : average waiting time across all lanes (seconds)
    avg_loss      : average training loss (None until buffer is ready)
    epsilon       : current exploration rate
    """

    def __init__(self, results_dir: str):
        self.log_dir   = os.path.join(results_dir, "logs")
        self.plot_dir  = os.path.join(results_dir, "plots")
        os.makedirs(self.log_dir,  exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

        self.csv_path  = os.path.join(self.log_dir, "training_log.csv")
        self.plot_path = os.path.join(self.plot_dir, "training.png")

        # In-memory history (lists grow each episode)
        self.episodes      = []
        self.total_rewards = []
        self.avg_waits     = []
        self.avg_losses    = []
        self.epsilons      = []

        # Write CSV header
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "total_reward", "avg_wait", "avg_loss", "epsilon"])

    # ── Public API ────────────────────────────────────────────────────────────

    def record(self, episode: int, total_reward: float,
               avg_wait: float, losses: list, epsilon: float):
        """
        Call once at the end of each episode.

        Parameters
        ----------
        episode      : current episode number (1-indexed)
        total_reward : sum of all rewards in the episode
        avg_wait     : mean waiting time across all decision steps
        losses       : list of loss values from training steps this episode
                       (can be empty if buffer wasn't ready yet)
        epsilon      : current epsilon value
        """
        avg_loss = float(np.mean(losses)) if losses else 0.0

        self.episodes     .append(episode)
        self.total_rewards.append(total_reward)
        self.avg_waits    .append(avg_wait)
        self.avg_losses   .append(avg_loss)
        self.epsilons     .append(epsilon)

        # Append row to CSV
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode, round(total_reward, 4),
                             round(avg_wait, 2), round(avg_loss, 6),
                             round(epsilon, 4)])

    def print_episode(self, episode: int, n_episodes: int):
        """Print a one-line summary for the latest episode."""
        print(
            f"  Episode {episode:>4}/{n_episodes} | "
            f"Reward: {self.total_rewards[-1]:>8.2f} | "
            f"Avg Wait: {self.avg_waits[-1]:>6.1f}s | "
            f"Loss: {self.avg_losses[-1]:.5f} | "
            f"eps: {self.epsilons[-1]:.3f}"
        )

    def plot(self):
        """
        Save a 4-panel PNG to results/plots/training.png.

        Panels
        ------
        1. Total reward per episode     — should trend upward
        2. Average waiting time         — should trend downward
        3. Training loss                — should decrease and stabilise
        4. Epsilon (exploration rate)   — steady decay curve
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle("DQN Training — Adaptive Traffic Signal Control", fontsize=14)

        data = [
            (self.total_rewards, "Total Reward",        "Episode", "Reward",       "tab:blue"),
            (self.avg_waits,     "Avg Waiting Time (s)","Episode", "Seconds",      "tab:red"),
            (self.avg_losses,    "Training Loss",       "Episode", "MSE Loss",     "tab:orange"),
            (self.epsilons,      "Epsilon (Exploration)","Episode","Epsilon",      "tab:green"),
        ]

        for ax, (values, title, xlabel, ylabel, color) in zip(axes.flat, data):
            ax.plot(self.episodes, values, color=color, linewidth=1.2, alpha=0.8)
            # Smoothed trend line (moving average over 10 episodes)
            if len(values) >= 10:
                smooth = np.convolve(values, np.ones(10)/10, mode="valid")
                ax.plot(self.episodes[9:], smooth, color=color,
                        linewidth=2.5, linestyle="--", label="10-ep avg")
                ax.legend(fontsize=8)
            ax.set_title(title, fontsize=11)
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=150)
        plt.close()
        print(f"  [Plot] Saved → {self.plot_path}")
