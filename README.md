# Adaptive Traffic Signal Control using Deep Reinforcement Learning

A DQN (Deep Q-Network) agent that learns to control traffic signals at a 4-way intersection using the [SUMO](https://eclipse.dev/sumo/) traffic simulator — outperforming traditional fixed-timing signals by **39% fewer halting vehicles** and **68% less waiting time**.

---

## Results

| Metric | DQN Agent (AI) | Fixed Timing | Improvement |
|--------|---------------|--------------|-------------|
| Avg Halting Vehicles | 5.6 | 9.2 | **39% better** |
| Avg Waiting Time | 53s | 168s | **68% better** |
| Peak Halting Vehicles | 17 | 25 | **32% better** |

Training progress over 500 episodes:

| Metric | Episode 1 | Episode 500 |
|--------|-----------|-------------|
| Avg Waiting Time | 234s | 39s |
| Total Reward | -4.62 | -1.79 |

---

## How It Works

The DQN agent observes the intersection every 10 simulation seconds and decides which signal phase to activate. It is rewarded for reducing the number of halting vehicles, and penalised for congestion.

- **State**: 32-dimensional vector — queue lengths and cumulative waiting times across all 16 incoming lanes, normalised to [0, 1]
- **Actions**: 4 discrete signal phases — NS straight, NS left turn, EW straight, EW left turn
- **Reward**: `-total_halting_vehicles / (MAX_QUEUE × N_LANES)` — bounded in [-1, 0]
- **Algorithm**: DQN with experience replay (50k buffer), target network (synced every 50 steps), Huber loss, epsilon-greedy exploration

---

## Requirements

| Tool | Version |
|------|---------|
| Python | 3.9+ |
| SUMO | 1.18+ |
| PyTorch | 2.0+ |

Verify SUMO is installed and on your PATH:
```bash
sumo --version
```

---

## Setup

**1. Clone the repository**
```bash
git clone https://github.com/wolfykulfi/Adaptive-Traffic-Signal-Control.git
cd Adaptive-Traffic-Signal-Control
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install Python dependencies**
```bash
pip install -r requirements.txt
```

**4. Verify TraCI is accessible**
```bash
python -c "import traci; print('TraCI OK')"
```

> If TraCI is not found, add the SUMO tools folder to your Python path:
> ```bash
> # Windows
> set PYTHONPATH=C:\Program Files (x86)\Eclipse\Sumo\tools
>
> # macOS / Linux
> export PYTHONPATH=/usr/share/sumo/tools
> ```

---

## Usage

> **Windows users**: always use the `-X utf8` flag to avoid encoding errors.

### Train the DQN agent
Runs 500 episodes (~30 minutes). Saves model checkpoints every 50 episodes to `models/`.
```bash
python -X utf8 scripts/train.py
```

### Evaluate — side-by-side visual comparison
Opens **two SUMO-GUI windows simultaneously** — the trained DQN agent on one side and a fixed-timing signal on the other, both running the same traffic scenario.
```bash
python -X utf8 scripts/evaluate.py
```

At the end of the 30-minute simulation, a comparison table is printed to the terminal with halting vehicle counts, waiting times, and improvement percentages.

### Generate result plots
Produces three PNG charts saved to `results/plots/`:
- `training_curves.png` — reward, waiting time, loss, and epsilon over 500 episodes
- `comparison.png` — DQN vs Fixed-Timing across time-series and bar charts
- `summary.png` — clean presentation chart with improvement annotations
```bash
python -X utf8 scripts/plot_results.py
```

### Inspect the intersection manually
```bash
sumo-gui --configuration-file sumo_configs/intersection.sumocfg
```

---

## Project Structure

```
ATSC/
├── src/
│   ├── environment/
│   │   └── sumo_env.py          # TraCI wrapper — state, action, reward
│   ├── agent/
│   │   ├── network.py           # QNetwork: 32 → 256 → 256 → 128 → 4
│   │   ├── replay_buffer.py     # Circular replay buffer (50k)
│   │   └── dqn_agent.py         # Epsilon-greedy, Huber loss, target network
│   └── utils/
│       └── metrics.py           # CSV logging and plot generation
├── configs/
│   └── train_config.py          # All hyperparameters in one place
├── scripts/
│   ├── train.py                 # Main training loop
│   ├── evaluate.py              # Side-by-side DQN vs Fixed-Timing evaluation
│   └── plot_results.py          # Generates all result plots
├── sumo_configs/
│   ├── intersection.net.xml     # Generated SUMO network (actuated TLS)
│   ├── intersection.rou.xml     # Probabilistic traffic flows (3 vehicle types)
│   ├── intersection.sumocfg     # Simulation config for DQN
│   ├── intersection_fixed.net.xml   # Fixed 35s green cycle network
│   └── intersection_fixed.sumocfg  # Simulation config for fixed-timing
├── models/
│   └── dqn_final.pth            # Trained model weights (500 episodes)
├── results/
│   ├── logs/training_log.csv    # Per-episode metrics
│   └── plots/                   # Output charts
└── requirements.txt
```

---

## Intersection Design

- **Type**: 4-way signalised, single junction
- **Arms**: North, South, East, West — each 500m
- **Lanes**: 4 per direction (16 incoming lanes total)
  - Lane 0: right turn
  - Lane 1 & 2: straight
  - Lane 3: left turn
- **Traffic**: Probabilistic flows with 3 vehicle types (cars, trucks, motorcycles). West approach is heaviest (~560 vehicles/hr), East is lightest (~160 vehicles/hr).
- **Signal phases**: NS straight → NS left → EW straight → EW left

---

## DQN Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate | 5e-4 |
| Discount factor (γ) | 0.99 |
| Epsilon start / end | 1.0 → 0.05 |
| Epsilon decay | 0.998 per episode |
| Replay buffer size | 50,000 |
| Batch size | 128 |
| Target network update | every 50 gradient steps |
| Episodes | 500 |
| Hidden layer size | 256 |

---

## Tech Stack

- **Python 3.11**
- **PyTorch** — neural network and training
- **SUMO + TraCI** — traffic simulation and real-time control
- **NumPy / Matplotlib** — data processing and visualisation
