# Reinforcement Learning for Cartpole Balancing in Isaac Sim

This repository presents a comparative study of four fundamental reinforcement learning (RL) algorithms—**Monte Carlo**, **Q-Learning**, **Double Q-Learning**, and **SARSA**—applied to the classical **Cartpole balancing problem** in NVIDIA Isaac Sim. The objective is to train an agent to balance a pole mounted on a cart by applying discrete forces within a specified range.

---

## 🧠 Algorithms Implemented

The following tabular RL algorithms were implemented and evaluated:

1. **Monte Carlo (MC) Control**  
   - Episodic, on-policy learning.
   - Updates action-values using full episode returns.
   - Exploration via epsilon-greedy policy.

2. **Q-Learning**  
   - Off-policy, temporal-difference learning.
   - Uses the maximum estimated future value for updates.

3. **Double Q-Learning**  
   - Variation of Q-Learning that reduces overestimation bias.
   - Maintains two Q-tables (`Q<sub>A</sub>`, `Q<sub>B</sub>`) and updates one using the other for evaluation.

4. **SARSA (State-Action-Reward-State-Action)**  
   - On-policy, temporal-difference learning.
   - Uses the actual action taken (under policy) for bootstrapping.

All algorithms share the same discretized state representation and discrete action space for consistency in evaluation.

---

## 🎯 Problem Description

The **Cartpole problem** involves a pole attached to a cart that moves along a one-dimensional track. The task is to apply a force to the cart in order to keep the pole upright.

- **State space** (continuous, discretized):
  - Cart position  ($x$)
  - Cart velocity
  - Pole angle
  - Pole angular velocity

- **Action space** (discrete):
  - 7 possible actions uniformly spanning the range [-15.0, 15.0] Newtons

---

## 🏗️ Environment Setup

### Requirements

- [Isaac Sim 2023.x](https://developer.nvidia.com/isaac-sim)
- Python ≥ 3.8
- NumPy
- [RL_Algorithm](/CartPole_4.2.0/RL_Algorithm/Algorithm/) module containing implementations of the RL agents

### Running the Training

To train each algorithm, head to [train.py](/CartPole_4.2.0/scripts/RL_Algorithm/train.py) or [train_sarsa.py](/CartPole_4.2.0/scripts/RL_Algorithm/train_sarsa.py). If the desired algorithm is MC, Q Learning, Double Q learning; change the `<algorithm name>` inside the `train.py` file in the following lines.

```py
Algorithm_name = "<algorithm name>"
    agent = <algorithm name>(...
```
To MC, Q_Learning or Double_Q_Learning.

To train SARSA algorithm, leave the code as is and run the `train_sarsa.py` file in the terminal instead.

To train, either use `train.py` or `train_sarsa.py` in the code below.
```bash
python scripts/RL_Algorithm/train.py --task Stabilize-Isaac-Cartpole-v0
```

Run this code in a seperate window to log the training result.
```bash
python -m tensorboard.main --logdir logs/sb3/Stabilize-Isaac-Cartpole-v0
```

---

## 🔬 Experimenting

This is the **baseline** configurations of 9 Hyperparameters used in this experiment.

```py
num_of_action = 9
action_range = [-15.0, 15.0]
discretize_state_weight = [10, 10, 2, 2]
learning_rate = 0.5
n_episodes = 10000
start_epsilon = 1.0
epsilon_decay = 0.9997
final_epsilon = 0.05
discount = 0.9
```
 The goal is to understand how each parameters interacts within each algorithms and determine which algorithm is best suited for this cartpole problem.

 Although all 9 of these parameters are related to the algorithm, the time takes for each algorithm is significant. So we have introduced a `preset parameters` referencing from previous researches favoring each algorithms to better see each algorithm's edges.

 ### Hyperparameter Presets for Algorithm Comparison

| Preset Name        | `learning_rate` | `discount` | `epsilon_decay` | Purpose                                                    |
|--------------------|------------------|------------|------------------|------------------------------------------------------------|
| **MC-Friendly**     | 0.5              | 0.99       | 0.9999           | Allows full episode exploration, suitable for delayed rewards |
| **Q-Learning Boost**| 0.7              | 0.9        | 0.9997           | Emphasizes fast learning and stable value updates          |
| **SARSA-Safe**      | 0.3              | 0.8        | 0.999            | Encourages smooth exploration decay for on-policy learning |

---

We will conduct 3 presets x 4 algorithms = 12 experiments, each giving comparative insight without deep tuning.

---

## Results

