"""
plot_results.py -- Generate all result plots
---------------------------------------------
Produces 3 figures saved to results/plots/:

    1. training_curves.png  -- 4-panel learning progress over 500 episodes
    2. comparison.png       -- DQN vs Fixed-Timing across every metric
    3. summary.png          -- Single clean bar chart for presentation

Run from project root with venv active:
    python -X utf8 scripts/plot_results.py
"""

import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH  = os.path.join(BASE_DIR, "results", "logs",  "training_log.csv")
PLOT_DIR  = os.path.join(BASE_DIR, "results", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Evaluation results (from evaluate.py run) ─────────────────────────────────
EVAL = {
    "dqn":   {"avg_halt": 5.6,  "avg_wait": 53.0,  "peak_halt": 17},
    "fixed": {"avg_halt": 9.2,  "avg_wait": 168.3, "peak_halt": 25},
}

# Time-series snapshots printed during evaluation (every 5 min)
EVAL_TIMELINE = {
    "time":       [300,  600,  900,  1200, 1500, 1800],
    "dqn_halt":   [4,    6,    8,    3,    8,    5   ],
    "fix_halt":   [2,    4,    11,   5,    12,   11  ],
    "dqn_wait":   [23,   97,   19,   14,   55,   24  ],
    "fix_wait":   [5,    59,   133,  53,   196,  269 ],
}

DQN_COLOR   = "#2196F3"   # blue
FIXED_COLOR = "#F44336"   # red
SMOOTH_WIN  = 20          # moving-average window for training curves


def moving_avg(data, w):
    return np.convolve(data, np.ones(w) / w, mode="valid")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  TRAINING CURVES
# ─────────────────────────────────────────────────────────────────────────────
def plot_training_curves():
    episodes, rewards, waits, losses, epsilons = [], [], [], [], []
    with open(LOG_PATH) as f:
        for row in csv.DictReader(f):
            episodes.append(int(row["episode"]))
            rewards .append(float(row["total_reward"]))
            waits   .append(float(row["avg_wait"]))
            losses  .append(float(row["avg_loss"]))
            epsilons.append(float(row["epsilon"]))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("DQN Training Progress  —  500 Episodes", fontsize=15, fontweight="bold")

    panels = [
        (axes[0,0], rewards,  "Total Reward per Episode",       "Reward",       "#9C27B0", "Higher is better"),
        (axes[0,1], waits,    "Avg Waiting Time per Episode",   "Seconds (s)",  "#F44336", "Lower is better"),
        (axes[1,0], losses,   "Training Loss (Huber)",          "Loss",         "#FF9800", "Should stabilise"),
        (axes[1,1], epsilons, "Epsilon (Exploration Rate)",     "Epsilon",      "#4CAF50", "Decays over time"),
    ]

    for ax, values, title, ylabel, color, note in panels:
        eps = np.array(episodes)
        vals = np.array(values)

        # Raw data (faint)
        ax.plot(eps, vals, color=color, alpha=0.25, linewidth=0.8)

        # Smoothed line
        if len(vals) >= SMOOTH_WIN:
            sm = moving_avg(vals, SMOOTH_WIN)
            ax.plot(eps[SMOOTH_WIN-1:], sm, color=color, linewidth=2.2, label=f"{SMOOTH_WIN}-ep avg")
            ax.legend(fontsize=8, loc="upper right")

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Episode", fontsize=9)
        ax.set_ylabel(ylabel,    fontsize=9)
        ax.text(0.02, 0.05, note, transform=ax.transAxes,
                fontsize=8, color="grey", style="italic")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, max(episodes))

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [1/3] Saved training_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  COMPARISON PLOTS  (multi-panel)
# ─────────────────────────────────────────────────────────────────────────────
def plot_comparison():
    t   = EVAL_TIMELINE["time"]
    dh  = EVAL_TIMELINE["dqn_halt"]
    fh  = EVAL_TIMELINE["fix_halt"]
    dw  = EVAL_TIMELINE["dqn_wait"]
    fw  = EVAL_TIMELINE["fix_wait"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("DQN Agent vs Fixed-Timing  —  Evaluation Comparison",
                 fontsize=14, fontweight="bold")

    dqn_patch   = mpatches.Patch(color=DQN_COLOR,   label="DQN Agent (AI)")
    fixed_patch = mpatches.Patch(color=FIXED_COLOR, label="Fixed-Timing")

    # ── Panel 1: Halting vehicles over time ───────────────────────────────
    ax = axes[0, 0]
    ax.plot(t, dh, "o-", color=DQN_COLOR,   linewidth=2, markersize=6, label="DQN Agent")
    ax.plot(t, fh, "s-", color=FIXED_COLOR, linewidth=2, markersize=6, label="Fixed Timing")
    ax.fill_between(t, dh, fh, where=[f > d for d, f in zip(dh, fh)],
                    alpha=0.15, color="green", label="DQN advantage")
    ax.set_title("Halting Vehicles Over Time", fontsize=11, fontweight="bold")
    ax.set_xlabel("Simulation Time (s)"); ax.set_ylabel("Halting Vehicles")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # ── Panel 2: Waiting time over time ───────────────────────────────────
    ax = axes[0, 1]
    ax.plot(t, dw, "o-", color=DQN_COLOR,   linewidth=2, markersize=6, label="DQN Agent")
    ax.plot(t, fw, "s-", color=FIXED_COLOR, linewidth=2, markersize=6, label="Fixed Timing")
    ax.fill_between(t, dw, fw, where=[f > d for d, f in zip(dw, fw)],
                    alpha=0.15, color="green")
    ax.set_title("Total Waiting Time Over Time", fontsize=11, fontweight="bold")
    ax.set_xlabel("Simulation Time (s)"); ax.set_ylabel("Waiting Time (s)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # ── Panel 3: Average metrics bar chart ────────────────────────────────
    ax = axes[1, 0]
    metrics = ["Avg Halting\nVehicles", "Avg Waiting\nTime (s)", "Peak Halting\nVehicles"]
    dqn_vals   = [EVAL["dqn"]["avg_halt"],   EVAL["dqn"]["avg_wait"],   EVAL["dqn"]["peak_halt"]]
    fixed_vals = [EVAL["fixed"]["avg_halt"], EVAL["fixed"]["avg_wait"], EVAL["fixed"]["peak_halt"]]

    x = np.arange(len(metrics))
    w = 0.35
    bars_dqn   = ax.bar(x - w/2, dqn_vals,   w, color=DQN_COLOR,   alpha=0.85, label="DQN Agent")
    bars_fixed = ax.bar(x + w/2, fixed_vals, w, color=FIXED_COLOR, alpha=0.85, label="Fixed Timing")

    for bar in bars_dqn:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in bars_fixed:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_title("Average Metrics Comparison", fontsize=11, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylabel("Value"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

    # ── Panel 4: Improvement % bar chart ─────────────────────────────────
    ax = axes[1, 1]
    imp_labels = ["Halting Vehicles\nReduced", "Waiting Time\nReduced", "Peak Halting\nReduced"]
    improvements = [
        (EVAL["fixed"]["avg_halt"]  - EVAL["dqn"]["avg_halt"])  / EVAL["fixed"]["avg_halt"]  * 100,
        (EVAL["fixed"]["avg_wait"]  - EVAL["dqn"]["avg_wait"])  / EVAL["fixed"]["avg_wait"]  * 100,
        (EVAL["fixed"]["peak_halt"] - EVAL["dqn"]["peak_halt"]) / EVAL["fixed"]["peak_halt"] * 100,
    ]
    colors = ["#4CAF50" if v > 0 else "#F44336" for v in improvements]
    bars = ax.bar(imp_labels, improvements, color=colors, alpha=0.85, edgecolor="white", linewidth=1.2)

    for bar, val in zip(bars, improvements):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5 if val >= 0 else bar.get_height() - 2,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("DQN Improvement over Fixed-Timing (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Improvement (%)"); ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(improvements) * 1.25)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [2/3] Saved comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SUMMARY PLOT  (clean single chart for presentation)
# ─────────────────────────────────────────────────────────────────────────────
def plot_summary():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Adaptive Traffic Signal Control  —  DQN vs Fixed-Timing",
                 fontsize=14, fontweight="bold", y=1.02)

    metrics = [
        ("Avg Halting Vehicles",   EVAL["dqn"]["avg_halt"],   EVAL["fixed"]["avg_halt"],   "Vehicles"),
        ("Avg Waiting Time",       EVAL["dqn"]["avg_wait"],   EVAL["fixed"]["avg_wait"],   "Seconds (s)"),
        ("Peak Halting Vehicles",  EVAL["dqn"]["peak_halt"],  EVAL["fixed"]["peak_halt"],  "Vehicles"),
    ]

    for ax, (title, dqn_val, fixed_val, unit) in zip(axes, metrics):
        bars = ax.bar(["DQN\nAgent", "Fixed\nTiming"],
                      [dqn_val, fixed_val],
                      color=[DQN_COLOR, FIXED_COLOR],
                      width=0.5, alpha=0.9,
                      edgecolor="white", linewidth=1.5)

        # Value labels on bars
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() * 1.02,
                    f"{bar.get_height():.1f}",
                    ha="center", va="bottom", fontsize=13, fontweight="bold")

        # Improvement arrow annotation
        imp = (fixed_val - dqn_val) / fixed_val * 100
        ax.annotate(f"{imp:.1f}% better",
                    xy=(0, dqn_val), xytext=(0.5, (dqn_val + fixed_val) / 2),
                    fontsize=10, color="#4CAF50", fontweight="bold", ha="center",
                    arrowprops=dict(arrowstyle="<->", color="#4CAF50", lw=1.5))

        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.set_ylabel(unit, fontsize=9)
        ax.set_ylim(0, fixed_val * 1.35)
        ax.grid(True, alpha=0.3, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Legend
    dqn_patch   = mpatches.Patch(color=DQN_COLOR,   label="DQN Agent (AI Control)")
    fixed_patch = mpatches.Patch(color=FIXED_COLOR, label="Fixed-Timing (Traditional)")
    fig.legend(handles=[dqn_patch, fixed_patch], loc="lower center",
               ncol=2, fontsize=10, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.08))

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [3/3] Saved summary.png")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating plots...")
    plot_training_curves()
    plot_comparison()
    plot_summary()
    print(f"\nAll plots saved to: {PLOT_DIR}")
    print("  - training_curves.png  : 500-episode learning progress")
    print("  - comparison.png       : DQN vs Fixed across all metrics")
    print("  - summary.png          : Clean presentation chart")
