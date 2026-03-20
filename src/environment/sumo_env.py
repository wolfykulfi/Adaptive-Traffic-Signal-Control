"""
sumo_env.py
-----------
TraCI-based SUMO environment for the Adaptive Traffic Signal Control agent.

Observation space  : per-lane queue length + waiting time for all 16 lanes
                     (4 approaches × 4 lanes each) → 32-dim float32 vector

Action space       : discrete, 4 green phases
                     0 – North/South straight
                     1 – North/South left turn
                     2 – East/West straight
                     3 – East/West left turn

Reward             : −Δ(total cumulative waiting time) per decision step
                     positive when congestion drops, negative when it grows
"""

import os
import sys
import numpy as np
import traci

# ── Phase definitions ────────────────────────────────────────────────────────
# Map action index → (phase_index_in_net, description)
# Phase indices must match the order netconvert wrote into intersection.net.xml.
# Typical order for a 4-approach actuated TLS:
#   0  NS straight green   2  EW straight green
#   1  NS straight yellow  3  EW straight yellow
# We insert protected left turns as additional phases if needed.
# The agent only picks GREEN phases; yellows are inserted automatically.

PHASE_NS_STRAIGHT = 0
PHASE_NS_LEFT     = 2
PHASE_EW_STRAIGHT = 4
PHASE_EW_LEFT     = 6

# Yellow phases sit one index above their corresponding green phase
YELLOW_DURATION = 4   # seconds

GREEN_PHASES = [
    PHASE_NS_STRAIGHT,
    PHASE_NS_LEFT,
    PHASE_EW_STRAIGHT,
    PHASE_EW_LEFT,
]

N_ACTIONS = len(GREEN_PHASES)

# ── Lane layout ──────────────────────────────────────────────────────────────
# 4 approaches × 4 lanes each (lane 0 = rightmost in direction of travel)
INCOMING_EDGES = ["N2C", "S2C", "E2C", "W2C"]
LANES_PER_EDGE = 4                          # must match numLanes in edg.xml

# Maximum values used for normalisation
MAX_QUEUE   = 40.0   # vehicles
MAX_WAIT    = 300.0  # seconds

# ── Decision step ────────────────────────────────────────────────────────────
DECISION_STEP   = 10   # agent acts every 10 simulation seconds
MIN_GREEN_TIME  = 10   # minimum green seconds before a phase change is allowed


class SumoEnv:
    """
    Gym-style wrapper around a SUMO simulation controlled via TraCI.

    Parameters
    ----------
    cfg_path   : str  – absolute path to intersection.sumocfg
    gui        : bool – launch sumo-gui instead of headless sumo
    max_steps  : int  – maximum simulation steps per episode (seconds)
    seed       : int  – random seed for reproducibility
    """

    def __init__(
        self,
        cfg_path: str,
        gui: bool = False,
        max_steps: int = 3600,
        seed: int = 42,
    ):
        self.cfg_path  = cfg_path
        self.gui       = gui
        self.max_steps = max_steps
        self.seed      = seed

        self.tls_id    = "C"           # traffic-light ID (matches node id)
        self.sumo_cmd  = None
        self._build_cmd()

        # Runtime state
        self._step        = 0          # simulation second
        self._decision    = 0          # decision step counter
        self._current_phase = PHASE_NS_STRAIGHT
        self._phase_timer   = 0        # seconds on current green phase
        self._prev_wait     = 0.0      # total waiting time at last decision

        # Observation / action dimensions (used by the DQN)
        self.state_size  = LANES_PER_EDGE * len(INCOMING_EDGES) * 2  # 32
        self.action_size = N_ACTIONS

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _build_cmd(self):
        binary = "sumo-gui" if self.gui else "sumo"
        self.sumo_cmd = [
            binary,
            "-c", self.cfg_path,
            "--random",                          # stochastic routes
            "--seed", str(self.seed),
            "--no-step-log", "true",
            "--waiting-time-memory", "3600",     # track full-episode waits
            "--time-to-teleport", "300",
        ]

    def _get_lanes(self):
        """Return list of all controlled lane IDs (incoming only)."""
        return [
            f"{edge}_{i}"
            for edge in INCOMING_EDGES
            for i in range(LANES_PER_EDGE)
        ]

    # ── TraCI data collectors ────────────────────────────────────────────────

    def _get_queue(self, lane: str) -> float:
        return traci.lane.getLastStepHaltingNumber(lane)

    def _get_wait(self, lane: str) -> float:
        return traci.lane.getWaitingTime(lane)

    def _total_waiting_time(self) -> float:
        return sum(self._get_wait(l) for l in self._get_lanes())

    # ── Gym interface ────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """Start a new episode. Returns the initial state vector."""
        if traci.isLoaded():
            traci.close()

        traci.start(self.sumo_cmd)

        self._step          = 0
        self._decision      = 0
        self._phase_timer   = 0
        self._current_phase = PHASE_NS_STRAIGHT
        self._prev_wait     = 0.0

        # Apply initial phase
        traci.trafficlight.setPhase(self.tls_id, self._current_phase)

        # Warm-up: run a few steps so lanes have vehicles before first decision
        for _ in range(DECISION_STEP):
            traci.simulationStep()
            self._step += 1

        self._prev_wait = self._total_waiting_time()
        return self._observe()

    def step(self, action: int):
        """
        Apply action, advance simulation by DECISION_STEP seconds.

        Returns
        -------
        state  : np.ndarray  – new observation
        reward : float
        done   : bool
        info   : dict
        """
        assert 0 <= action < N_ACTIONS, f"Invalid action {action}"

        desired_phase = GREEN_PHASES[action]

        # Insert yellow transition if phase is changing
        if desired_phase != self._current_phase and self._phase_timer >= MIN_GREEN_TIME:
            yellow_phase = self._current_phase + 1
            traci.trafficlight.setPhase(self.tls_id, yellow_phase)
            for _ in range(YELLOW_DURATION):
                traci.simulationStep()
                self._step += 1

            self._current_phase = desired_phase
            self._phase_timer   = 0

        # Apply (or keep) the green phase
        traci.trafficlight.setPhase(self.tls_id, self._current_phase)

        # Advance simulation for the full decision step
        for _ in range(DECISION_STEP):
            traci.simulationStep()
            self._step      += 1
            self._phase_timer += 1

        # Compute reward
        # Use negative total halting vehicles (bounded in [-1, 0])
        # This is a direct, stable signal: fewer stopped cars = higher reward
        lanes = self._get_lanes()
        total_halting = sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)
        reward = -total_halting / (MAX_QUEUE * len(lanes))

        current_wait = self._total_waiting_time()

        state = self._observe()
        done  = self._step >= self.max_steps

        info = {
            "step":         self._step,
            "total_wait":   current_wait,
            "total_halting": total_halting,
            "phase":        self._current_phase,
            "queue":        self._queue_per_lane(),
        }

        self._decision += 1
        return state, reward, done, info

    def close(self):
        """Shut down TraCI / SUMO."""
        if traci.isLoaded():
            traci.close()

    # ── Observation builder ──────────────────────────────────────────────────

    def _observe(self) -> np.ndarray:
        """
        Build a normalised 32-dim state vector:
            [queue_lane0, ..., queue_lane15,
             wait_lane0,  ..., wait_lane15]
        """
        lanes = self._get_lanes()
        queues = np.array([self._get_queue(l) for l in lanes], dtype=np.float32)
        waits  = np.array([self._get_wait(l)  for l in lanes], dtype=np.float32)

        queues = np.clip(queues / MAX_QUEUE, 0.0, 1.0)
        waits  = np.clip(waits  / MAX_WAIT,  0.0, 1.0)

        return np.concatenate([queues, waits])

    def _queue_per_lane(self) -> dict:
        return {l: self._get_queue(l) for l in self._get_lanes()}

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def current_step(self) -> int:
        return self._step

    @property
    def current_phase(self) -> int:
        return self._current_phase
