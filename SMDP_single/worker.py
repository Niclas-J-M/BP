# Import necessary libraries
import torch
import torch.optim as optim
from SMDP_single.networks import PolicyNetwork, ValueNetwork
from utils.prioritized_memory import Memory
import numpy as np

class Worker:
    def __init__(self, input_dim, output_dim, device, gamma=0.99):
        """
        Initialize the Worker agent with policy and value networks, optimizers, and memory buffer.

        Parameters:
        - input_dim: Dimension of the input state.
        - output_dim: Dimension of the output actions.
        - device: Device to run the computations on (CPU or GPU).
        - gamma: Discount factor for future rewards.
        """
        self.device = device
        self.policy_net = PolicyNetwork(input_dim, output_dim).to(device)
        self.value_net = ValueNetwork(input_dim).to(device)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0007)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=0.0007)
        self.memory = Memory(10000)  # Capacity of the replay buffer
        self.batch_size = 512
        self.gamma = gamma
        self.value_loss_weight = 0.01
        self.policy_losses = []
        self.value_losses = []
        self.entropy_coeff = 0.01

    def train(self, transitions):
        """ 
        On-policy training method using collected transitions.

        Parameters:
        - transitions: A list of transition tuples (state, action, reward, next_state, done).
        """
        # Unpack transitions
        batch = list(zip(*transitions))
        states = torch.tensor(np.vstack(batch[0]), dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch[1], dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(batch[2], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.vstack(batch[3]), dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch[4], dtype=torch.float32, device=self.device).unsqueeze(1)

        # Predict values for current and next states
        predicted_values = self.value_net(states)
        next_values = self.value_net(next_states)

        # Calculate target and advantage (TD error)
        td_targets = rewards + self.gamma * next_values * (1 - dones)
        td_errors = td_targets - predicted_values

        # Update value network
        value_loss = td_errors.pow(2).mean()
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Update policy network
        log_probs = torch.log(self.policy_net(states).gather(1, actions))
        policy_loss = -(log_probs * td_errors.detach()).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Store experiences in the replay buffer for Self-Imitation Learning (SIL)
        for i in range(len(transitions)):
            self.memory.add(td_errors[i].item(), (batch[0][i], batch[1][i], batch[2][i], batch[3][i], batch[4][i]))

    def train_sil(self):
        """ 
        Self-Imitation Learning (SIL) from the memory buffer.
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

        # Update policy network using SIL
        probs = self.policy_net(states)
        log_probs = torch.log(probs.gather(1, actions) + epsilon)
        policy_loss = -(log_probs * rewards * is_weights).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Update priorities in the memory based on the new policy
        new_td_errors = rewards - self.value_net(states)
        for idx, error in zip(idxs, new_td_errors):
            self.memory.update([idx], [error.item()])

    def select_action(self, state):
        """
        Select an action based on the current policy.

        Parameters:
        - state: The current state of the agent.

        Returns:
        - The action selected by the policy.
        """
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            probs = self.policy_net(state)
            dist = torch.distributions.Categorical(probs)
        return dist.sample().item()
