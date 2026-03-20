"""
evaluate.py -- Side-by-side evaluation: DQN Agent vs Fixed-Timing
------------------------------------------------------------------
Runs two simultaneous SUMO simulations:
    Window  : DQN agent controlling the signal in SUMO-GUI (visual)
    Headless: Fixed-timing simulation running in background

Metrics from both are collected and compared at the end.

Run from project root with venv active:
    python -X utf8 scripts/evaluate.py
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import traci
from configs.train_config import CFG
from src.agent.dqn_agent import DQNAgent
from src.environment.sumo_env import (
    GREEN_PHASES, INCOMING_EDGES, LANES_PER_EDGE,
    PHASE_NS_STRAIGHT, YELLOW_DURATION, DECISION_STEP,
    MIN_GREEN_TIME, MAX_QUEUE,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DQN_CFG       = os.path.join(BASE_DIR, "sumo_configs", "intersection.sumocfg")
FIXED_CFG     = os.path.join(BASE_DIR, "sumo_configs", "intersection_fixed.sumocfg")
MODEL         = os.path.join(BASE_DIR, "models", "dqn_final.pth")

TLS_ID     = "C"
EVAL_STEPS = 1800
SEED       = 42
LANES      = [f"{e}_{i}" for e in INCOMING_EDGES for i in range(LANES_PER_EDGE)]

COMMON_FLAGS = [
    "--seed",                str(SEED),
    "--no-step-log",         "true",
    "--waiting-time-memory", "3600",
    "--time-to-teleport",    "300",
]


def observe(conn) -> np.ndarray:
    queues = np.array([conn.lane.getLastStepHaltingNumber(l) for l in LANES], dtype=np.float32)
    waits  = np.array([conn.lane.getWaitingTime(l)           for l in LANES], dtype=np.float32)
    return np.concatenate([
        np.clip(queues / MAX_QUEUE, 0.0, 1.0),
        np.clip(waits  / 300.0,    0.0, 1.0),
    ])


def get_metrics(conn):
    halting = sum(conn.lane.getLastStepHaltingNumber(l) for l in LANES)
    waiting = sum(conn.lane.getWaitingTime(l)           for l in LANES)
    return halting, waiting


def evaluate():
    print("=" * 65)
    print("  DQN Agent vs Fixed-Timing  |  Side-by-Side Evaluation")
    print("=" * 65)

    # ── Load model ────────────────────────────────────────────────────────
    agent = DQNAgent(
        state_size  = CFG["state_size"],
        action_size = CFG["action_size"],
        hidden_size = CFG["hidden_size"],
    )
    agent.load(MODEL)
    agent.epsilon = 0.0
    print("  Model loaded  |  epsilon = 0 (pure greedy policy)\n")

    # ── Start DQN simulation in SUMO-GUI (visible) ────────────────────────
    print("  Opening DQN agent simulation window...")
    dqn_cmd = ["sumo-gui", "-c", DQN_CFG,
               "--start", "true", "--delay", "75"] + COMMON_FLAGS
    traci.start(dqn_cmd, label="dqn", numRetries=60)
    print("  DQN window connected.")

    # Wait for first window to fully initialise before opening the second
    time.sleep(3)

    # ── Start Fixed simulation in SUMO-GUI (visible) ─────────────────────
    print("  Opening fixed-timing simulation window...")
    fix_cmd = ["sumo-gui", "-c", FIXED_CFG,
               "--start", "true", "--delay", "75"] + COMMON_FLAGS
    traci.start(fix_cmd, label="fixed", numRetries=60)
    print("  Fixed window connected.\n")

    # ── Inject label POIs — ID is used as the on-map display text ────────
    traci.switch("dqn")
    traci.poi.add("DQN Agent  [AI Control]", -460, 460,
                  color=(0, 200, 255, 255), layer=200)

    traci.switch("fixed")
    traci.poi.add("Fixed-Timing Signal  [Traditional]", -460, 460,
                  color=(255, 80, 80, 255), layer=200)

    # Set initial DQN phase
    traci.switch("dqn")
    traci.trafficlight.setPhase(TLS_ID, PHASE_NS_STRAIGHT)

    # ── Metrics ───────────────────────────────────────────────────────────
    dqn_halting, fix_halting = [], []
    dqn_waiting, fix_waiting = [], []

    current_phase = PHASE_NS_STRAIGHT
    phase_timer   = 0
    step          = 0

    print(f"  Running {EVAL_STEPS}s evaluation ({EVAL_STEPS//60} min)...")
    print(f"  {'Time':>6}  {'DQN Halt':>9}  {'Fixed Halt':>10}  {'DQN Wait':>9}  {'Fixed Wait':>10}")
    print("  " + "-" * 52)

    while step < EVAL_STEPS:

        # ── DQN phase decision ─────────────────────────────────────────────
        if step % DECISION_STEP == 0 and step > 0:
            traci.switch("dqn")
            action  = agent.select_action(observe(traci.getConnection("dqn")))
            desired = GREEN_PHASES[action]

            if desired != current_phase and phase_timer >= MIN_GREEN_TIME:
                traci.trafficlight.setPhase(TLS_ID, current_phase + 1)
                for _ in range(YELLOW_DURATION):
                    traci.switch("dqn");   traci.simulationStep()
                    traci.switch("fixed"); traci.simulationStep()
                    step += 1
                current_phase = desired
                phase_timer   = 0
                traci.switch("dqn")
            traci.trafficlight.setPhase(TLS_ID, current_phase)

        # ── Advance both ───────────────────────────────────────────────────
        traci.switch("dqn");   traci.simulationStep()
        traci.switch("fixed"); traci.simulationStep()
        step        += 1
        phase_timer += 1

        # ── Record every 10 steps ──────────────────────────────────────────
        if step % DECISION_STEP == 0:
            traci.switch("dqn")
            dh = sum(traci.lane.getLastStepHaltingNumber(l) for l in LANES)
            dw = sum(traci.lane.getWaitingTime(l) for l in LANES)
            traci.switch("fixed")
            fh = sum(traci.lane.getLastStepHaltingNumber(l) for l in LANES)
            fw = sum(traci.lane.getWaitingTime(l) for l in LANES)

            dqn_halting.append(dh); dqn_waiting.append(dw)
            fix_halting.append(fh); fix_waiting.append(fw)

            if step % 300 == 0:
                print(f"  {step:>5}s  {dh:>9.0f}  {fh:>10.0f}  {dw:>9.1f}  {fw:>10.1f}")

    # ── Close ─────────────────────────────────────────────────────────────
    traci.switch("dqn");   traci.close()
    traci.switch("fixed"); traci.close()

    # ── Results ───────────────────────────────────────────────────────────
    dah = np.mean(dqn_halting); fah = np.mean(fix_halting)
    daw = np.mean(dqn_waiting); faw = np.mean(fix_waiting)
    halt_imp = (fah - dah) / max(fah, 1) * 100
    wait_imp = (faw - daw) / max(faw, 1) * 100

    print("\n" + "=" * 65)
    print("  FINAL RESULTS")
    print("=" * 65)
    print(f"  {'Metric':<32} {'DQN Agent':>12}  {'Fixed':>10}")
    print("  " + "-" * 58)
    print(f"  {'Avg Halting Vehicles':<32} {dah:>12.1f}  {fah:>10.1f}")
    print(f"  {'Avg Waiting Time (s)':<32} {daw:>12.1f}  {faw:>10.1f}")
    print(f"  {'Peak Halting Vehicles':<32} {max(dqn_halting):>12.0f}  {max(fix_halting):>10.0f}")
    print("  " + "-" * 58)
    print(f"  {'Halting Reduction':<32} {halt_imp:>11.1f}%")
    print(f"  {'Waiting Time Reduction':<32} {wait_imp:>11.1f}%")
    print("=" * 65)
    if halt_imp > 0:
        print(f"\n  DQN reduced congestion by {halt_imp:.1f}% vs fixed timing.")
    else:
        print(f"\n  Fixed timing outperformed DQN by {abs(halt_imp):.1f}% — more training needed.")


if __name__ == "__main__":
    evaluate()
