import torch
import torch.nn as nn

class DynamicPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, device):
        super(DynamicPolicyNetwork, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU()
        )
        self.heads = nn.ModuleList()
        self.action_dim = action_dim
        self.device = device

    def add_head(self):
        head = nn.Sequential(
            nn.Linear(64, self.action_dim),
            nn.Softmax(dim=-1)
        ).to(self.device)
        self.heads.append(head)

    def forward(self, x, option_idx):
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
        super(DynamicValueNetwork, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU()
        )
        self.heads = nn.ModuleList()
        self.device = device

    def add_head(self):
        head = nn.Linear(64, 1).to(self.device)
        self.heads.append(head)

    def forward(self, x, option_idx):
        shared_out = self.shared_layers(x)
        value = self.heads[option_idx](shared_out)
        return value
