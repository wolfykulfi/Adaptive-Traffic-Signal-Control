# Viva Preparation Guide
## Adaptive Traffic Signal Control — DQN Minor Project

---

## Understand the Core Idea First

Be able to explain the project in 2 sentences without technical jargon:

> "I built an AI agent that controls traffic signals at an intersection. Instead of using fixed
> timers, it watches the traffic in real time and decides which direction should get the green
> light to reduce waiting time — and it learns to do this better over thousands of simulations."

If you can say that confidently, you've already won half the viva.

---

## Know These Concepts Cold

These are what examiners ask in every RL project viva:

### Reinforcement Learning Basics
- What is an agent, environment, state, action, reward?
- What is the difference between exploration and exploitation?
- What is epsilon-greedy and why does epsilon decay?

### DQN Specifically
- Why do you need a neural network?
  → Because the state space is too large for a Q-table
- What is the replay buffer and why does it exist?
  → Breaks correlation between consecutive experiences, stabilises training
- Why do you have two networks (policy + target)?
  → If you use one network for both prediction and target, the target keeps shifting
     like chasing a moving goal. The target network is frozen for stability.
- What is the Bellman equation?
  → Q(s,a) = r + gamma * max Q(s',a') — current reward plus discounted future reward

### Your Specific Project
- What is the state?
  → 32 numbers — queue length and waiting time on each of the 16 lanes, normalised to 0-1
- What are the 4 actions?
  → NS straight green, NS left green, EW straight green, EW left green
- What is the reward?
  → Negative number of halting vehicles — the agent learns to minimise congestion
- What is SUMO?
  → Open-source traffic simulator used by research labs and transport authorities worldwide
- What is TraCI?
  → Python API that lets your code talk to SUMO in real time — read sensor data, set signal phases

---

## Prepare for These Likely Questions

**"Why DQN and not a simpler algorithm?"**
> The state has 32 continuous values — a traditional Q-table would need to store a row for
> every possible combination, which is computationally impossible. A neural network
> approximates the Q-function across the continuous state space.

**"Why not just use a longer green phase or actuated signals?"**
> Actuated signals react to one direction at a time. The DQN agent considers the full state
> of all 16 lanes simultaneously and makes a globally informed decision — it can choose to
> hold a green phase or switch based on the complete picture.

**"How do you know the agent actually learned something?"**
> Two ways — first, the training curve shows average waiting time dropping from 234s to 39s
> over 500 episodes. Second, the side-by-side evaluation against a fixed-timing baseline on
> the same traffic shows 68% less waiting time and 39% fewer halting vehicles.

**"What is the discount factor gamma and why 0.99?"**
> Gamma controls how much future rewards matter. At 0.99 the agent is very forward-looking
> — a reward 100 steps ahead is still worth 0.99^100 = 37% of its face value. This is
> appropriate for traffic because clearing congestion now affects flow for minutes afterward.

**"What would you improve?"**
> Multiple seeds for statistical validation, Double DQN to reduce Q-value overestimation,
> and extending it to a network of connected intersections rather than a single one.

---

## What to Show During the Demo

Run this order:

1. Open results/plots/training_curves.png
   - Show reward improving and waiting time dropping over 500 episodes
   - Talk about epsilon decaying as agent shifts from random to learned behaviour

2. Open results/plots/summary.png
   - Clean bar chart, easy to read, strong visual impact
   - Let the numbers speak

3. Run the live evaluation (if they allow it):
      python -X utf8 scripts/evaluate.py
   - Two windows side by side
   - Point at the fixed-timing window building up queues while DQN clears them faster

4. Show the code briefly
   - Open src/agent/dqn_agent.py, point to the Bellman update (lines 156-158)
   - Shows you understand what the code does, not just that it runs

---

## The Three Things That Will Impress an Examiner

1. You can explain WHY each design choice was made, not just what it is
2. You know what the LIMITATIONS are (single seed, single intersection)
3. You can point to the training curve and explain what was happening at different stages
   - Early episodes: agent is acting randomly (epsilon = 1.0)
   - Mid-training: agent starts finding patterns, reward improves
   - Late training: agent stabilises, loss flattens

---

---

## DQN Deep Explanation

### Start with Q-Learning

Before DQN, there was Q-Learning. The idea is simple — build a table where every row is a
state and every column is an action. Each cell stores a number called a Q-value, which means:

  "If I am in this state and take this action, how much total reward will I get in the future?"

The agent always picks the action with the highest Q-value. Over time it updates this table
using the Bellman equation:

  Q(state, action) = reward  +  gamma x max Q(next_state, all_actions)
                     ↑                   ↑
               what I got now    best I can get from here onward

The problem: your state is 32 continuous numbers. A table would need billions of rows.
Impossible.

---

### The "Deep" Part — Replace the Table with a Neural Network

Instead of a table, train a neural network that takes a state as input and outputs a Q-value
for every action:

  Input (32 numbers)  →  [256] → [256] → [128]  →  Output (4 Q-values)
    queue lengths                                     one per signal phase
    waiting times

The network learns to approximate the Q-table across the entire continuous state space.
That is the "Deep" in Deep Q-Network.

---

### Three Problems DQN Solves

PROBLEM 1 — Consecutive experiences are correlated

If you train step-by-step, each sample looks almost identical to the previous one (traffic
changes slowly). The network overfits to recent experience and forgets everything else.

Fix — Replay Buffer:
Store the last 50,000 experiences (state, action, reward, next_state) in a buffer.
Sample a random batch of 128 at each training step. Now each batch contains experiences
from different times and different traffic conditions — the network learns from a diverse mix.

  self.buffer = deque(maxlen=50000)       # circular, old experiences get dropped
  batch = random.sample(self.buffer, 128) # random mix every training step

---

PROBLEM 2 — The training target keeps moving

To train the network you need a target value (the correct answer). The Bellman target is:

  target = reward + gamma x max Q(next_state)

But Q(next_state) comes from the same network you are updating. Every time you update the
network, the target changes too — like trying to hit a moving bullseye. Training becomes
unstable.

Fix — Target Network:
Keep two identical networks. The policy network is trained every step. The target network
is a frozen copy that only gets updated every 50 steps. The target network provides stable
Q-value estimates while the policy network learns.

  self.policy_net = QNetwork(...)   # updated every step
  self.target_net = QNetwork(...)   # frozen copy, synced every 50 steps

  max_next_q = self.target_net(next_states).max(dim=1)[0]      # stable reference
  target_q   = rewards + 0.99 * max_next_q * (1 - dones)       # Bellman target

---

PROBLEM 3 — Early on the agent knows nothing, so it needs to explore

If the agent always picks the highest Q-value from the start, it gets stuck doing the same
few actions it tried first. It never discovers better strategies.

Fix — Epsilon-Greedy Exploration:
With probability epsilon, pick a random action. Otherwise pick the best known action.
Start at epsilon = 1.0 (fully random) and decay every episode until epsilon = 0.05
(mostly learned policy).

  if random() < epsilon:
      return random_action()     # explore
  else:
      return argmax(Q_values)    # exploit what we know

In the training log you can see epsilon going from 1.0 down to ~0.05 over 500 episodes.

---

### The Full Training Loop (One Step)

  1. Agent observes state s
     → 16 queue lengths + 16 waiting times = 32 numbers, all normalised to [0,1]

  2. Epsilon-greedy → pick action a (one of 4 signal phases)

  3. SUMO advances 10 seconds with that phase active

  4. Observe reward r = -halting_vehicles / (MAX_QUEUE x N_LANES)
     Negative so minimising halting = maximising reward. Range [-1, 0].

  5. Observe next state s'

  6. Store (s, a, r, s', done) in replay buffer

  7. Sample random batch of 128 from buffer

  8. For each sample compute Bellman target:
     y = r  +  0.99 x max Q_target(s')

  9. Compute loss = Huber( y,  Q_policy(s, a) )

  10. Backpropagate, update policy network weights

  11. Every 50 steps: copy policy weights → target network

---

### Why Huber Loss and Not MSE?

MSE squares the error. If a Q-value prediction is far off (error = 100), MSE gives a
gradient of 10,000 — the weights get a massive update and training explodes.

Huber loss behaves like MSE for small errors but like absolute error for large ones —
capping the gradient and keeping training stable. This is why the loss stayed flat at
~0.00016 throughout training instead of exploding.

---

### Why Gamma = 0.99?

Gamma controls how much future rewards matter.

  Reward 10 steps ahead  is worth: 0.99^10  = 0.90  (90% of face value)
  Reward 50 steps ahead  is worth: 0.99^50  = 0.61  (61% of face value)
  Reward 100 steps ahead is worth: 0.99^100 = 0.37  (37% of face value)

Traffic congestion has long-term effects — clearing a queue now keeps the intersection
flowing for the next several minutes. A high gamma makes the agent plan ahead rather
than just reacting to the immediate situation.

---

### What the Agent Actually Learned

By episode 500, the agent has discovered things like:
- If the West approach (heaviest traffic) has long queues, prioritise EW straight green
- Do not switch phases too often — yellow phases waste 4 seconds each time
- Hold a green phase longer when queues are still clearing
- A light queue on one lane does not justify disrupting a heavy flow on another

It did not learn these rules explicitly — they emerged from:
  500 episodes  x  360 decisions per episode  =  180,000 decision steps of trial and error

---

### Neural Network Architecture

  Layer         Size    Activation
  -----------   -----   ----------
  Input         32      —
  Hidden 1      256     ReLU
  Hidden 2      256     ReLU
  Hidden 3      128     ReLU
  Output        4       — (raw Q-values, one per action)

ReLU (Rectified Linear Unit) = max(0, x). It outputs 0 for negative inputs and passes
positive inputs through unchanged. Used because it is simple, fast, and avoids the
vanishing gradient problem that sigmoid/tanh suffer from in deep networks.

---

### Key Numbers to Remember

  Training episodes    : 500
  Decisions per episode: ~360  (3600s / 10s decision step)
  Total decisions      : ~180,000
  State dimensions     : 32
  Actions              : 4
  Replay buffer size   : 50,000
  Batch size           : 128
  Learning rate        : 0.0005
  Gamma                : 0.99
  Epsilon start → end  : 1.0 → 0.05
  Target sync interval : every 50 gradient steps
  Training time        : ~30 minutes on CPU

---

## One-Line Answers for Quick Questions

| Question                        | Answer                                                              |
|---------------------------------|---------------------------------------------------------------------|
| What problem does it solve?     | Reduces traffic congestion at intersections using AI                |
| What algorithm?                 | Deep Q-Network (DQN) — Mnih et al., 2015 (DeepMind)               |
| What simulator?                 | SUMO (Simulation of Urban Mobility) — open source, DLR Germany     |
| How is it trained?              | 500 simulated episodes, each 1 hour of traffic                      |
| How long did training take?     | ~30 minutes on CPU                                                  |
| State size?                     | 32 (16 queue lengths + 16 waiting times, all normalised)            |
| Action space?                   | 4 discrete signal phases                                            |
| Reward?                         | -halting_vehicles / (MAX_QUEUE x N_LANES), range [-1, 0]           |
| Key result?                     | 68% less waiting time vs fixed-timing signal                        |
| Neural network architecture?    | 32 -> 256 -> 256 -> 128 -> 4 (fully connected, ReLU activations)   |
