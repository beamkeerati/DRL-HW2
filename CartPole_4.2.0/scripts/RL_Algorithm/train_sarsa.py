"""Script to train RL agent using SARSA."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
import numpy as np

from omni.isaac.lab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import SARSA algorithm and other algorithms if needed
from RL_Algorithm.Algorithm.SARSA import SARSA 
from RL_Algorithm.Algorithm.Q_Learning import Q_Learning
from RL_Algorithm.Algorithm.Double_Q_Learning import Double_Q_Learning
from RL_Algorithm.Algorithm.MC import MC
from tqdm import tqdm

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from datetime import datetime
import random
from torch.utils.tensorboard import SummaryWriter  # TensorBoard logging

from omni.isaac.lab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml
from omni.isaac.lab_tasks.utils import get_checkpoint_path
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

# Import extensions to set up environment tasks
import CartPole.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with an on-policy SARSA agent."""
    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # directory for logging into
    log_dir = os.path.join("logs", "sb3", args_cli.task, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    
    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # ==================================================================== #
    # ========================= Agent Initialization ===================== #

    #Fixed by experiment
    num_of_action = 10
    action_range = [-15.0, 15.0]  # [min, max]
    n_episodes = 1000
    start_epsilon = 1.0
    
    #Tuning parameters
    discretize_state_weight = [1, 3, 1, 1]  # [pose_cart:int, pose_pole:int, vel_cart:int, vel_pole:int]
    learning_rate = 0.25#0.5
    epsilon_decay = 0.997 #0.9997  # reduce the exploration over time
    final_epsilon = 0.05
    discount = 0.9  # Set discount to a valid float (e.g., 0.99)
    
    task_name = str(args_cli.task).split('-')[0]  # e.g., Stabilize, SwingUp

    # --- Modified: Use SARSA with proper on-policy update ---
    Algorithm_name = "SARSA"
    agent = SARSA(
        num_of_action=num_of_action,
        action_range=action_range,
        discretize_state_weight=discretize_state_weight,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        discount_factor=discount
    )

    # reset environment and choose initial action
    obs, _ = env.reset()
    action, action_idx = agent.get_action(obs)
    timestep = 0
    sum_reward = 0

    # simulate environment using the SARSA sequence:
    # 1. Initialize state and action.
    # 2. For each step, take the chosen action, observe reward and next state.
    # 3. Choose the next action from next state.
    # 4. Update Q(s,a) using: Q(s,a) <- Q(s,a) + α*(r + γ*Q(s',a') - Q(s,a))
    for episode in tqdm(range(n_episodes)):
        # For each episode, reset environment and get initial action
        obs, _ = env.reset()
        action, action_idx = agent.get_action(obs)
        done = False
        cumulative_reward = 0

        while not done:
            next_obs, reward, terminated, truncated, _ = env.step(action)
            cumulative_reward += reward.item()
            done = terminated or truncated

            if not done:
                next_action, next_action_idx = agent.get_action(next_obs)
            else:
                next_action, next_action_idx = None, None

            # Update SARSA: pass the next action index explicitly to the update rule.
            agent.update(obs, action_idx, reward.item(), next_obs, next_action_idx, done)

            # Update state and action for next iteration
            obs = next_obs
            action = next_action
            action_idx = next_action_idx

        sum_reward += cumulative_reward

        # Log per-episode metrics to TensorBoard
        writer.add_scalar("Episode/Reward", cumulative_reward, episode)
        writer.add_scalar("Episode/Epsilon", agent.epsilon, episode)
        if len(agent.q_values) > 0:
            q_vals = np.array(list(agent.q_values.values()))
            writer.add_histogram("Q_values", q_vals, episode)

        # Every 100 episodes, print average reward and run an evaluation episode
        if episode % 100 == 0 and episode != 0:
            print("avg_score: ", sum_reward / 100.0)
            sum_reward = 0

            # Save SARSA agent
            q_value_file = f"{Algorithm_name}_{episode}_{num_of_action}_{action_range[1]}_{discretize_state_weight[0]}_{discretize_state_weight[1]}.json"
            full_path = os.path.join(f"q_value/{task_name}", Algorithm_name)
            agent.save_q_value(full_path, q_value_file)

            # Deployment evaluation: run one episode with greedy policy (epsilon set to final value)
            saved_epsilon = agent.epsilon
            agent.epsilon = final_epsilon
            eval_obs, _ = env.reset()
            eval_done = False
            eval_reward = 0
            eval_action, eval_action_idx = agent.get_action(eval_obs)
            while not eval_done:
                eval_next_obs, eval_r, eval_term, eval_trunc, _ = env.step(eval_action)
                eval_reward += eval_r.item()
                eval_done = eval_term or eval_trunc
                if not eval_done:
                    eval_action, eval_action_idx = agent.get_action(eval_next_obs)
                eval_obs = eval_next_obs
            writer.add_scalar("Deployment/EvalReward", eval_reward, episode)
            agent.epsilon = saved_epsilon

        agent.decay_epsilon()

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

    print("!!! Training is complete !!!")
    writer.close()
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
