# Import necessary libraries
import torch
import torch.nn as nn

class DynamicPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, device):
        """
        Initialize the Dynamic Policy Network.

        Parameters:
        - state_dim: Dimension of the input state.
        - action_dim: Dimension of the output actions.
        - device: Device to run the computations on (CPU or GPU).
        """
        super(DynamicPolicyNetwork, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU()
        )
        self.heads = nn.ModuleList()  # List of policy heads
        self.action_dim = action_dim
        self.device = device

    def add_head(self):
        """
        Add a new head to the policy network.
        """
        head = nn.Sequential(
            nn.Linear(64, self.action_dim),
            nn.Softmax(dim=-1)
        ).to(self.device)
        self.heads.append(head)

    def forward(self, x, option_idx):
        """
        Forward pass through the Dynamic Policy Network.

        Parameters:
        - x: Input state tensor.
        - option_idx: Index of the option head to use.

        Returns:
        - probs: Output action probabilities.
        """
        shared_out = self.shared_layers(x)
        probs = self.heads[option_idx](shared_out)
        
        # Cap probabilities to avoid very low and very high values
        min_prob = 1e-8
        max_prob = 1.0 - 1e-8
        probs = torch.clamp(probs, min=min_prob, max=max_prob)
        probs = probs / probs.sum(dim=-1, keepdim=True)  # Re-normalize to ensure they sum to 1
        
        return probs

class DynamicValueNetwork(nn.Module):
    def __init__(self, state_dim, device):
        """
        Initialize the Dynamic Value Network.

        Parameters:
        - state_dim: Dimension of the input state.
        - device: Device to run the computations on (CPU or GPU).
        """
        super(DynamicValueNetwork, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU()
        )
        self.heads = nn.ModuleList()  # List of value heads
        self.device = device

    def add_head(self):
        """
        Add a new head to the value network.
        """
        head = nn.Linear(64, 1).to(self.device)
        self.heads.append(head)

    def forward(self, x, option_idx):
        """
        Forward pass through the Dynamic Value Network.

        Parameters:
        - x: Input state tensor.
        - option_idx: Index of the option head to use.

        Returns:
        - value: Output value tensor.
        """
        shared_out = self.shared_layers(x)
        value = self.heads[option_idx](shared_out)
        return value
