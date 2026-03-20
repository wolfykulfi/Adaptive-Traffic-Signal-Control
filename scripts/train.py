"""
train.py  —  Main DQN Training Loop
------------------------------------
Run from the project root with the venv active:

    python scripts/train.py

What happens:
    • A SUMO simulation runs headlessly for each episode (1 hour simulated).
    • Every 10 simulated seconds the DQN agent observes the intersection,
      picks a signal phase, receives a reward, and stores the experience.
    • After collecting enough experiences the neural network starts training.
    • Every 10 episodes the model is saved and a plot is updated.
"""

import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.train_config    import CFG
from src.environment.sumo_env import SumoEnv
from src.agent.dqn_agent      import DQNAgent
from src.utils.metrics        import MetricsTracker


def train():
    print("=" * 60)
    print("  Adaptive Traffic Signal Control — DQN Training")
    print("=" * 60)
    print(f"  Episodes    : {CFG['n_episodes']}")
    print(f"  Sim length  : {CFG['max_steps']}s per episode")
    print(f"  Batch size  : {CFG['batch_size']}")
    print(f"  Epsilon     : {CFG['epsilon_start']} -> {CFG['epsilon_end']}")
    print("=" * 60)

    # ── Initialise environment ────────────────────────────────────────────────
    env = SumoEnv(
        cfg_path  = CFG["sumo_cfg"],
        gui       = CFG["gui"],
        max_steps = CFG["max_steps"],
    )

    # ── Initialise agent ──────────────────────────────────────────────────────
    agent = DQNAgent(
        state_size      = CFG["state_size"],
        action_size     = CFG["action_size"],
        hidden_size     = CFG["hidden_size"],
        lr              = CFG["lr"],
        gamma           = CFG["gamma"],
        epsilon_start   = CFG["epsilon_start"],
        epsilon_end     = CFG["epsilon_end"],
        epsilon_decay   = CFG["epsilon_decay"],
        buffer_capacity = CFG["buffer_capacity"],
        batch_size      = CFG["batch_size"],
        target_update   = CFG["target_update"],
    )

    # ── Metrics tracker ───────────────────────────────────────────────────────
    tracker = MetricsTracker(results_dir=CFG["results_dir"])

    # ── Training loop ─────────────────────────────────────────────────────────
    for episode in range(1, CFG["n_episodes"] + 1):

        state       = env.reset()     # start new simulation, get first state
        done        = False
        ep_reward   = 0.0
        ep_waits    = []              # average wait per decision step
        ep_losses   = []              # loss per training step

        # ── Episode loop (one decision every DECISION_STEP seconds) ───────
        while not done:

            # 1. Choose action (phase) — random or greedy
            action = agent.select_action(state)

            # 2. Apply action to SUMO, advance simulation, get feedback
            next_state, reward, done, info = env.step(action)

            # 3. Store this experience in replay buffer
            agent.store(state, action, reward, next_state, done)

            # 4. Train on a random batch (returns None until buffer is ready)
            loss = agent.train()
            if loss is not None:
                ep_losses.append(loss)

            # 5. Move to next state
            state      = next_state
            ep_reward += reward
            ep_waits  .append(info["total_wait"])

        # ── End of episode ────────────────────────────────────────────────
        avg_wait = sum(ep_waits) / len(ep_waits) if ep_waits else 0.0

        # Decay epsilon (agent explores less as it learns more)
        agent.decay_epsilon()

        # Record and print stats
        tracker.record(episode, ep_reward, avg_wait, ep_losses, agent.epsilon)
        tracker.print_episode(episode, CFG["n_episodes"])

        # Save model checkpoint and updated plot every N episodes
        if episode % CFG["save_every"] == 0:
            model_path = os.path.join(CFG["model_dir"], f"dqn_ep{episode}.pth")
            agent.save(model_path)
            tracker.plot()

    # ── Training complete ─────────────────────────────────────────────────────
    env.close()

    # Final save
    agent.save(os.path.join(CFG["model_dir"], "dqn_final.pth"))
    tracker.plot()

    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  Model  → {CFG['model_dir']}/dqn_final.pth")
    print(f"  Plot   → {CFG['results_dir']}/plots/training.png")
    print(f"  Log    → {CFG['results_dir']}/logs/training_log.csv")
    print("=" * 60)


if __name__ == "__main__":
    train()
