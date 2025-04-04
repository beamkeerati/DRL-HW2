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
   - Maintains two Q-tables (`Q_A`, `Q_B`) and updates one using the other for evaluation.

4. **SARSA (State-Action-Reward-State-Action)**  
   - On-policy, temporal-difference learning.
   - Uses the actual action taken (under policy) for bootstrapping.

All algorithms share the same discretized state representation and discrete action space for consistency in evaluation.

---

## 🎯 Problem Description

The **Cartpole problem** involves a pole attached to a cart that moves along a one-dimensional track. The task is to apply a force to the cart in order to keep the pole upright.

- **State space** (continuous, discretized):
  - Cart position  ($x$)
  - Cart velocity ($v$)
  - Pole angle ($\theta$)
  - Pole angular velocity ($\omega$)

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
to MC, Q_Learning or Double_Q_Learning.

--- 

Starts with launching the `Isaaclab` environment

```bash
conda activate isaaclab
```

then browse to your workspace.

```bash
cd DRL-HW2/CartPole_4.2.0/
```

To train SARSA algorithm, leave the code as is and run the `train_sarsa.py` file in the terminal instead.

Otherwise, either use `train.py` or `train_sarsa.py` in the code below.
```bash
python scripts/RL_Algorithm/train.py --task Stabilize-Isaac-Cartpole-v0
```

With the same directory, run this code in a seperate window to log the training result.
```bash
python -m tensorboard.main --logdir logs/sb3/Stabilize-Isaac-Cartpole-v0
```

---

## 🔬 Experimenting

This is a **fixed** configurations of 4 Hyperparameters used in this experiment. In essence, these parameters will not be changed.

1. **`num_of_action`**

```py
num_of_action = 10
```

Fixed to maintain a consistent, discrete representation of the action space that aligns with the physical dynamics of the environment.

---

2. **`action range`**

```py
action_range = [-15.0, 15.0]
```

The action range is particularly important in the swing-up phase of the cartpole, where a wider range is required to generate sufficient momentum. In contrast, during the stabilization phase, the cartpole operates closer to the upright position, so a more limited action range such as `[-15.0, 15.0]` is typically sufficient.


---

3. **`n_ episodes`**

```py
n_episodes = 1000
```

Standardized training duration, enabling fair performance comparisons and isolating the impact of other hyperparameters.

---

4. **`start_epsilon`**

```py
start_epsilon = 1.0
```

Fixed to establish a consistent exploratory baseline across experiments, ensuring early-stage exploration is sufficient and reproducible.

---
from [train.py](/CartPole_4.2.0/scripts/RL_Algorithm/train.py) 
 
   #hyperparameters  
   ~~num_of_action = 10~~  
   ~~action_range = [-15.0, 15.0]~~  
   discretize_state_weight = [10, 50, 10, 50]  
   learning_rate = 0.05  
   ~~n_episodes = 1000~~  
   ~~start_epsilon = 1.0~~  
   epsilon_decay = 0.99  
   final_epsilon = 0.05  
   discount = 0.9  

leaves us with only these tunable parameters:

```py
discretize_state_weight = [10, 10, 2, 2]  
learning_rate = 0.5  
epsilon_decay = 0.9997  
final_epsilon = 0.05  
discount = 0.9  
```

---

## Tuning `discretize_state_weight`

The `discretize_state_weight` variable determines how many bins are used to discretize each continuous state variable for the RL agent. Below is a table summarizing the parameters, their default values, candidate tuning ranges, and additional notes.

| State Variable        | Description                                           | Default Bins | Candidate Values     | Notes                                                                                          |
|-----------------------|-------------------------------------------------------|--------------|----------------------|------------------------------------------------------------------------------------------------|
| **Cart Position**     | Position of the cart along the track                  | 10           | 1, 2, 3, 5, 10, 20, 50     | Higher resolution can capture subtle changes but may increase the state space size.            |
| **Pole Angle**        | Angle of the pole relative to the vertical            | 10           | 1, 2, 3, 4, 5, 10, 20   | Critical for maintaining balance; finer bins may improve accuracy at a computational cost.     |
| **Cart Velocity**     | Speed of the cart                                     | 2            | 1, 2, 3, 5, 10, 20, 50               | Typically less sensitive; a coarse discretization is often sufficient.                         |
| **Pole Angular Velocity** | Speed of the pole's rotation                      | 2            | 1, 2, 3, 5, 10          | Similar to cart velocity; adjust only if finer detail is needed for stability.                 |

We will begin with these candidate values and perform a grid search to determine the optimal discretization for each state variable. Once we identify promising configurations, we will further fine-tune within the optimal range.


### Epsilon

First we try each parameters with only 1000 episodes and we find the optimal epsilon decay value.

- Purple: 0.9  
- Green: 0.99  
- Grey: 0.997
- Yellow: 0.999  
- Pink: 0.9997

![](/images/Tune_epsilon_value_1000.png)

Thus the epsilon decay value of 0.997 is the most optimal for 1000 episodes.

### Discrete pole angle weight

- Grey: 1 
- Green: 2
- Blue: 3
- Purple: 4
- Cyan: 5
- Pink: 10
- Yellow: 20

![](/images/Tune_angle_bin_value.png)

The value with highest reward is **3**, so it will be used for the next parameter. 

### Discrete cart position weight

- Yellow: 1 
- Purple: 2
- Green: 3
- Orange: 5
- Blue: 10
- Cyan: 20
- Pink: 50

![](/images/Tune_pose_bin_value.png)

The value with highest reward is **1**

### Discrete cart velocity weight

- Red: 1
- Yellow: 2 
- Purple: 3
- Green: 5
- Orange: 10
- Blue: 20
- Pink: 50

![](/images/Tune_cart_velocity_bin_value.png)

The value with highest reward is **1**

### Discrete cart velocity weight

- Grey: 1
- Red: 2 
- Cyan: 3
- Green: 5
- Orange: 10

![](/images/Tune_angular_vel_value.png)

The value with highest reward is **1**

the tuned value is:

```py
discretize_state_weight = [1, 3, 1, 1] 
```

---

### Coarse Learning rate

- Green: 0.01
- Orange: 0.05 
- Grey: 0.1
- Red: 0.5
- Pink: 1
- Yellow: 5

![](/images/Tune_learning_rate.png)

The coarse comparison indicates that the initial value of **0.5** is still the best value. We have to do fine comparison next to determine the true best value for this environment.

### Lower End Fine Learning rate

- Green: 0.2
- Purple: 0.25
- Orange: 0.3
- Grey: 0.35
- Cyan: 0.4
- Pink: 0.45
- Red: 0.5

![](/images/Tune_learning_rate_fine_lower.png)

We can till see fluctuations so before we conclude we will take a look at the Upper end first.

### Upper End Fine Learning rate

- Red: 0.5
- Yellow: 0.55
- Purple: 0.6
- Green: 0.65
- Orange: 0.7

![](/images/Tune_learning_rate_fine_upper.png)

The most promising value is still around **0.25** thus we will use that value for now.

the tuned value is:

```py
learning_rate = 0.25
```

---

**This is far from the ideal value and is definitely not the way to go for reinforcement learning. It's probably due to the entanglement of each parameters that is making our incremental experiment struggles. With this we will make do with what we have and compare the algorithms as is.**

---

## Comparison

- Monte Carlo: Purple
- Q Learning: Pink
- Double Q Learning: Yellow
- SARSA: Green

![](/images/1_3_1_1_Comparison.png)

Here we can see that the value that we have previously tuned does not work at all on other algorithms and is very dependent on one algorithm.

![](/images/All_Graphs.png)

And Increasing episode to 10000 doesn't help either.

## Result

1. Which algorithm performs best?  

Well, it's complicated. With all honesty, we didn't train enough times to find the cherry picked variable suited for our algorithms even with only 1000 episodes. The short answer is that the MC algorithm is the most flexible algorithm based on our results. But still, that's because it was tuned with MC only and that's why it's probably heavily biased towards the MC. With 9 parameters even with some fixed and stepping experiment we are unable to find the exact relationship of these values. It probably need either an extensive training session or a very lucky parameter picks to get the desired effect. It's unfortunate, but we do accept that we underestimate the time it takes to get through all of the trail and error of training. With the current data and time that we have we conclude that the Monte Carlo algorithm performs best only because it's the one we used to tuned the parameters.

2. Why does it perform better than the others?

Well, with the full implementation and correct parameter tuning, theoretically Double Q Learning should be the best overall since it can reduce overestimation bias. But with limited episodes and parameters oriented towards the MC that's why the current implementation favors the MC.

3. How do the resolutions of the action space and observation space affect the learning process? Why?

You probably notice our very low discrete state weight by now, when we try a higher number it just doesn't get any reward at all. The value that would make most sense would be an odd number so we have a middle region that the cartpole can stabilize in the middle. so I'd say [5, 13, 2, 2] which means we divide the range of [-15, 15] into 5 ranges and the pole angle into 13 regions. It's indeed a very critical variable but we are unable to make it work and find any working hyperparameters.

---

**What did we get from this experiment? How does each parameter affect the learning process?**

1. `num_of_action = 10`

This variable defines the number of discrete actions available to the agent. For example, with num_of_action = 10, the agent can choose from 10 different actions between the range of -15.0 and 15.0 (as specified by action_range).

A higher number of actions increases the granularity of control, allowing the agent to make finer adjustments. However, it also increases the size of the action space, which can slow down learning as the agent has to evaluate more actions.

- Increase the number (e.g., 20): Gives the agent more control but makes learning slower due to the larger action space.

- Decrease the number (e.g., 5): Speeds up learning but may reduce the agent's ability to make precise control adjustments.

---

2. `action_range = [-15.0, 15.0]`

Defines the range of possible action values the agent can choose from, where -15.0 is the minimum action and 15.0 is the maximum.
This defines the physical limits of the agent's actions, which directly affects how large the agent’s movement or control actions can be.

- Increase the range (e.g., [-20.0, 20.0]): The agent can perform stronger actions, which may be necessary for environments requiring large movements or adjustments.

- Decrease the range (e.g., [-10.0, 10.0]): Limits the control precision, potentially speeding up learning but reducing the agent's ability to perform larger corrective actions.

---

3. `n_episodes = 1000`

Defines the total number of episodes the agent will run during training.
More episodes provide more opportunities for the agent to explore and learn, but longer training can be time-consuming.

- Increase the number (e.g., 5000): Provides the agent more time to explore and learn, which generally leads to better performance, though it requires more computation time.

- Decrease the number (e.g., 500): Speeds up training but may result in less exploration and underfitting, potentially limiting the agent’s performance.

---

4. `start_epsilon = 1.0`
   
Defines the initial exploration rate (epsilon). A value of 1.0 means that the agent will initially explore randomly, and as it learns, it will exploit its knowledge more.
A high epsilon ensures that the agent explores the environment at the beginning, helping it discover all possible actions. Over time, epsilon decays to encourage the agent to exploit the learned policy.

- Increase the value (e.g., 1.2): Leads to more exploration in the early stages but may delay exploitation.

- Decrease the value (e.g., 0.5): Encourages more immediate exploitation of the policy, which could result in faster learning but might prevent the agent from fully exploring the environment.

---

5. `discretize_state_weight = [1, 3, 1, 1]`

Defines the number of bins used to discretize each of the four state variables (cart position, pole angle, cart velocity, and pole angular velocity).
The state space discretization affects how finely the agent perceives the environment. Finer discretization captures more details but increases the size of the state space, which can slow down learning.

- Increase the number (e.g., [10, 20, 5, 5]): Provides a more detailed representation of the state space, allowing the agent to learn more precise policies but increasing the learning time.

- Decrease the number (e.g., [1, 2, 1, 1]): Speeds up learning but may result in a less accurate policy due to coarse representation.

---

6. `learning_rate = 0.25`

Determines how much the Q-values are updated during each learning step. A higher learning rate means that updates to Q-values will be larger.
A higher learning rate can make learning faster but may lead to instability, while a lower learning rate promotes more stable updates but can result in slower convergence.

- Increase the value (e.g., 0.5): Speeds up learning but may cause instability, leading to fluctuating Q-values.

- Decrease the value (e.g., 0.1): Slows down learning but improves stability, allowing the agent to converge more smoothly.

---

7. `epsilon_decay = 0.997`

Controls the rate at which exploration (epsilon) decreases over time. A smaller value means the agent will decay its exploration rate more slowly, encouraging more exploration.
Slower epsilon decay means the agent will continue exploring for longer, which can be useful for environments where the agent needs more exploration. However, it may delay the transition to exploitation.

- Increase the value (e.g., 0.999): Slows down the decay, encouraging more exploration over a longer period, which can be useful if the environment is large or complex.

- Decrease the value (e.g., 0.95): Decays exploration more quickly, focusing the agent on exploitation sooner, which could lead to faster convergence but might miss better strategies.

---

8. `final_epsilon = 0.05`

Defines the minimum epsilon value after decay. This ensures that the agent will still have some level of exploration even after the decay process.
A lower final epsilon value results in more exploitation and less exploration as the training progresses, but still allows for some random exploration.

- Increase the value (e.g., 0.1): Leads to more exploration in the later stages of learning, which might help find better solutions at the cost of slower exploitation.

- Decrease the value (e.g., 0.01): Leads to faster convergence to an optimal policy but with very little further exploration.
  
---

9. `discount = 0.9`

Defines how much future rewards are considered when updating Q-values. A higher discount factor gives more weight to future rewards.
A high discount factor encourages the agent to consider long-term goals, while a lower value focuses on immediate rewards. This is crucial in environments where long-term planning is needed.

- Increase the value (e.g., 0.95): Makes the agent focus more on long-term rewards, which is useful for tasks where the agent needs to plan ahead.

- Decrease the value (e.g., 0.85): Encourages the agent to focus on immediate rewards, which can make learning faster but may prevent the agent from considering long-term consequences.

## Author

Keerati Ubonmart 65340500003   
Manaswin Anekvisudwong 65340500049

---