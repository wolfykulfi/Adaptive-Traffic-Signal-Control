"""
train_config.py  --  All hyperparameters in one place
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CFG = {

    # -- SUMO ------------------------------------------------------------------
    "sumo_cfg"   : os.path.join(BASE_DIR, "sumo_configs", "intersection.sumocfg"),
    "gui"        : False,
    "max_steps"  : 3600,          # 1 simulated hour per episode

    # -- Training --------------------------------------------------------------
    "n_episodes" : 500,           # was 100 -- needs more episodes to converge

    # -- DQN Agent -------------------------------------------------------------
    "state_size"      : 32,
    "action_size"     : 4,
    "hidden_size"     : 256,      # was 128 -- larger network
    "lr"              : 5e-4,     # was 1e-3 -- lower lr prevents Q divergence
    "gamma"           : 0.99,
    "epsilon_start"   : 1.0,
    "epsilon_end"     : 0.05,
    "epsilon_decay"   : 0.998,    # was 0.995 -- slower decay, more exploration
    "buffer_capacity" : 50_000,   # was 10,000 -- larger memory
    "batch_size"      : 128,      # was 64
    "target_update"   : 50,       # was 10 -- less frequent = more stable

    # -- Saving ----------------------------------------------------------------
    "save_every"  : 50,           # save checkpoint every 50 episodes
    "model_dir"   : os.path.join(BASE_DIR, "models"),
    "results_dir" : os.path.join(BASE_DIR, "results"),
}
