import pandas as pd
import numpy as np

# Plotting the mean and standard deviation
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
# file_path_nor = 'BP/total_rewards8_8.csv'
# file_path_head = 'BP/total_rewards8_8_head.csv'
#file_path_nor = 'BP/total_steps8_8.csv'
file_path_head = 'BP/total_rewards_16_head.csv'

#data_rewards = pd.read_csv(file_path_nor, header=None)
data_rewards_head = pd.read_csv(file_path_head, header=None)

# Convert the DataFrame to a numpy array
#data_rewards_np = data_rewards.to_numpy()
data_rewards_head_np = data_rewards_head.to_numpy()

# Compute the average reward for each step across all runs
#average_rewards = np.mean(data_rewards_np, axis=0)
average_rewards_head = np.mean(data_rewards_head_np, axis=0)

# Compute the standard deviation for each step across all runs
#std_dev = np.std(data_rewards_np, axis=0)
std_dev_head = np.std(data_rewards_head_np, axis=0)

#upper_bound = np.minimum(average_rewards + std_dev, 300)
#lower_bound = np.maximum(average_rewards - std_dev, 0)

upper_bound_head = np.minimum(average_rewards_head + std_dev_head, 300)
lower_bound_head = np.maximum(average_rewards_head - std_dev_head, 0)

# Plot for the first dataset
#plt.plot(average_rewards, label='HRL-SIL', color='red', linewidth=0.3)
#äplt.fill_between(range(len(average_rewards)), lower_bound, upper_bound, alpha=0.1, color='red')
plt.figure(figsize=(12,4))
# # Plot for the second dataset
plt.plot(average_rewards_head, label='HRL-SIL-Head', color='blue', linewidth=0.3)
plt.fill_between(range(len(average_rewards_head)), lower_bound_head, upper_bound_head, alpha=0.1, color='blue')

# Add labels and title
plt.xlabel('Episode')
plt.ylabel('Total Rewards')
plt.title('Total Averaged Rewards Over Episodes for HRL and HRL-Head')
plt.legend()
plt.grid(True)
plt.show()







