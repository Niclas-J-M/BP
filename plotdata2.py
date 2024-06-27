import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
file_path_nor = 'BP/total_rewards8_8.csv'
file_path_head = 'BP/total_rewards8_8_head.csv'

data_rewards = pd.read_csv(file_path_nor, header=None)
data_rewards_head = pd.read_csv(file_path_head, header=None)

# Convert the DataFrame to a numpy array
data_rewards_np = data_rewards.to_numpy()
data_rewards_head_np = data_rewards_head.to_numpy()

# Compute the average reward for each step across all runs
average_rewards = np.mean(data_rewards_np, axis=0)
average_rewards_head = np.mean(data_rewards_head_np, axis=0)

# Compute the standard deviation for each step across all runs
std_dev = np.std(data_rewards_np, axis=0)
std_dev_head = np.std(data_rewards_head_np, axis=0)

upper_bound = np.minimum(average_rewards + std_dev, 3)
lower_bound = np.maximum(average_rewards - std_dev, 0)

upper_bound_head = np.minimum(average_rewards_head + std_dev_head, 3)
lower_bound_head = np.maximum(average_rewards_head - std_dev_head, 0)

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot for the first dataset
ax1.plot(average_rewards, label='HRL', color='red', linewidth=0.8)
ax1.fill_between(range(len(average_rewards)), lower_bound, upper_bound, alpha=0.1, color='red')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Total Rewards')
ax1.set_title('Total Averaged Rewards for HRL each episode')
ax1.legend()
ax1.grid(True)

# Plot for the second dataset
ax2.plot(average_rewards_head, label='HRL-Head', color='blue', linewidth=0.8)
ax2.fill_between(range(len(average_rewards_head)), lower_bound_head, upper_bound_head, alpha=0.1, color='blue')
ax2.set_xlabel('Episode')
ax2.set_ylabel('Total Rewardss')
ax2.set_title('Total Averaged Rewards for HRL-Head each episode')
ax2.legend()
ax2.grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()
