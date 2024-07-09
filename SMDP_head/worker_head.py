# Import necessary libraries
import torch
import torch.optim as optim
from SMDP_head.shared_networks import DynamicPolicyNetwork, DynamicValueNetwork
from utils.prioritized_memory import Memory
import numpy as np

class Worker_Head:
    def __init__(self, input_dim, output_dim, device, gamma=0.99):
        """
        Initialize the Worker_Head agent with dynamic policy and value networks, optimizers, and memory buffer.

        Parameters:
        - input_dim: Dimension of the input state.
        - output_dim: Dimension of the output actions.
        - device: Device to run the computations on (CPU or GPU).
        - gamma: Discount factor for future rewards.
        """
        self.device = device
        self.policy_net = DynamicPolicyNetwork(input_dim, output_dim, device).to(device)
        self.value_net = DynamicValueNetwork(input_dim, device).to(device)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0007)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=0.0007)
        self.memory = Memory(10000)  # Capacity of the replay buffer
        self.batch_size = 512
        self.gamma = gamma
        self.value_loss_weight = 0.01
        self.policy_losses = []
        self.value_losses = []
        self.option_count = 0

    def add_option_head(self):
        """
        Add a new option head to the policy and value networks.

        Returns:
        - Index of the new option head.
        """
        self.policy_net.add_head()
        self.value_net.add_head()
        self.option_count += 1
        return self.option_count - 1

    def train(self, transitions, option_idx):
        """
        On-policy training method using collected transitions.

        Parameters:
        - transitions: A list of transition tuples (state, action, reward, next_state, done).
        - option_idx: Index of the option head to train.
        """
        batch = list(zip(*transitions))
        states = torch.tensor(np.vstack(batch[0]), dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch[1], dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(batch[2], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.vstack(batch[3]), dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch[4], dtype=torch.float32, device=self.device).unsqueeze(1)

        # Predict values
        predicted_values = self.value_net(states, option_idx)
        next_values = self.value_net(next_states, option_idx)

        # Calculate target and advantage
        td_targets = rewards + self.gamma * next_values * (1 - dones)
        td_errors = td_targets - predicted_values

        # Value network update
        value_loss = td_errors.pow(2).mean()
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Policy network update
        log_probs = torch.log(self.policy_net(states, option_idx).gather(1, actions))
        policy_loss = -(log_probs * td_errors.detach()).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Store experiences in the replay buffer for SIL
        for i in range(len(transitions)):
            self.memory.add(td_errors[i].item(), (batch[0][i], batch[1][i], batch[2][i], batch[3][i], batch[4][i]))

    def train_sil(self, option_idx):
        """
        Self-Imitation Learning (SIL) from the memory buffer.

        Parameters:
        - option_idx: Index of the option head to train.
        """
        if len(self.memory) < self.batch_size:
            return  # Not enough samples to perform a batch update

        transitions, idxs, is_weights = self.memory.sample(self.batch_size)
        epsilon = 1e-8  # Small value to add for numerical stability

        batch = list(zip(*transitions))
        states = torch.tensor(np.vstack(batch[0]), dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch[1], dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(batch[2], dtype=torch.float32, device=self.device).unsqueeze(1)
        is_weights = torch.tensor(is_weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Policy network update using SIL
        probs = self.policy_net(states, option_idx)
        log_probs = torch.log(probs.gather(1, actions) + epsilon)
        policy_loss = -(log_probs * rewards * is_weights).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Update priorities in the memory based on the new policy
        new_td_errors = rewards - self.value_net(states, option_idx)
        for idx, error in zip(idxs, new_td_errors):
            self.memory.update([idx], [error.item()])

    def select_action(self, state, option_idx):
        """
        Select an action based on the current policy.

        Parameters:
        - state: The current state of the agent.
        - option_idx: Index of the option head to use.

        Returns:
        - The action selected by the policy.
        """
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            probs = self.policy_net(state, option_idx)
            dist = torch.distributions.Categorical(probs)
        return dist.sample().item()
