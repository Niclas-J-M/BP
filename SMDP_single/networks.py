import torch
import torch.nn as nn
    
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        probs = self.layers(x)
        
        # Cap probabilities to avoid very low and very high values
        min_prob = 1e-8
        max_prob = 1.0 - 1e-8
        probs = torch.clamp(probs, min=min_prob, max=max_prob)
        probs = probs / probs.sum(dim=-1, keepdim=True)  # Re-normalize to ensure they sum to 1
        
        return probs

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.layers(x)
    
