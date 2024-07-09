# Import necessary libraries
import gymnasium as gym
from minigrid.wrappers import NESWActionsImage
from SMDP_single.SMPD import SMDP
from SMDP_naive.SMDP_naive import SMDP_naive
from SMDP_head.SMDP_head import SMDP_head
import numpy as np
from utils.utils import plot_rewards, plot_steps
from gymnasium.envs.registration import register
import torch
import csv

# Set the device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Ensure that PyTorch recognizes the GPU
if device.type == 'cuda':
    print(f"CUDA is available. GPU device name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Running on CPU.")

def main():
    # Create and configure the environment
    env = gym.make('MiniGrid-DoorKey-8x8-v0')
    # To use 16x16 Environment uncomment the following line:
    #env = gym.make('MiniGrid-DoorKey-16x16-v0')
    env = NESWActionsImage(env, 0, 10000)

    # Define experiment parameters
    num_regions = 4
    tasks = [0, 1, 2]
    num_actions = env.n_actions
    num_episodes = 1000
    all_rewards = []
    all_steps = []
    naive = False  # Set to True for SMDP_naive
    head = True    # Set to True for SMDP_head

    num_runs = 5  # Number of times to run the experiment

    # Run the experiment multiple times
    for _ in range(num_runs):
        # Choose the appropriate SMDP method based on the flags
        if naive:
            total_steps, total_rewards = SMDP_naive(env, num_regions, num_actions, num_episodes, tasks, device)
        elif head:
            total_steps, total_rewards = SMDP_head(env, num_regions, num_actions, num_episodes, tasks, device)
        else:
            total_steps, total_rewards = SMDP(env, num_regions, num_actions, num_episodes, tasks, device)

        # Print and collect the rewards and steps for each run
        print(total_rewards)
        all_rewards.append(total_rewards)
        all_steps.append(total_steps)
    
    # Compute average and standard deviation for rewards and steps
    avg_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)
    avg_steps = np.mean(all_steps, axis=0)
    std_steps = np.std(all_steps, axis=0)

    # Plot the results
    plot_steps(avg_steps, std_steps)
    plot_rewards(avg_rewards, std_rewards)

# Entry point of the script
if __name__ == "__main__":
    main()
