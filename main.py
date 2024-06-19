import gymnasium as gym
from minigrid.wrappers import NESWActionsImage
from SMPD import SMDP
import numpy as np
from utils.utils import plot_rewards, plot_steps
from gymnasium.envs.registration import register
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Ensure that PyTorch recognizes the GPU
if device.type == 'cuda':
    print(f"CUDA is available. GPU device name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Running on CPU.")


register(
    id='SimpleEnv-v0',
    entry_point='simple_env:SimpleEnv',
)

def main():
    env = gym.make('MiniGrid-DoorKey-16x16-v0')
    #register_env()
    #env = gym.make('SimpleEnv-v0')
    #env = gym.make('MiniGrid-DoorKey-8x8-v0')
    env = NESWActionsImage(env, 0, 10000)

    num_regions = 16
    tasks = [0, 1, 2]
    num_actions = env.n_actions
    num_episodes = 400
    all_rewards = []
    all_steps = []

    num_runs = 4
    for _ in range(num_runs):
        total_steps, total_rewards = SMDP(env, num_regions, num_actions, num_episodes, tasks, device)
        all_rewards.append(total_rewards)
        all_steps.append(total_steps)
    
    avg_rewards = np.mean(all_rewards, axis=0)
    avg_steps = np.mean(all_steps, axis=0)

    plot_steps(avg_steps)
    plot_rewards(avg_rewards)





if __name__ == "__main__":
    main()