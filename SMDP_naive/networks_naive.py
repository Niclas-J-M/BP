# Import necessary libraries
import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        Initialize the Policy Network.

        Parameters:
        - state_dim: Dimension of the input state.
        - action_dim: Dimension of the output actions.
        """
        super(PolicyNetwork, self).__init__()
        # Define the network layers
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        """
        Forward pass through the Policy Network.

        Parameters:
        - x: Input state tensor.

        Returns:
        - probs: Output action probabilities.
        """
        probs = self.layers(x)
        
        # Cap probabilities to avoid very low and very high values
        min_prob = 1e-8
        max_prob = 1.0 - 1e-8
        probs = torch.clamp(probs, min=min_prob, max=max_prob)
        probs = probs / probs.sum(dim=-1, keepdim=True)  # Re-normalize to ensure they sum to 1
        
        return probs

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        """
        Initialize the Value Network.

        Parameters:
        - state_dim: Dimension of the input state.
        """
        super(ValueNetwork, self).__init__()
        # Define the network layers
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """
        Forward pass through the Value Network.

        Parameters:
        - x: Input state tensor.

        Returns:
        - Output value tensor.
        """
        return self.layers(x)
