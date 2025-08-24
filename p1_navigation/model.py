import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Value network: maps state -> action values (Q(s, a))."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """
        Args:
            state_size (int): dimension of state vector
            action_size (int): number of discrete actions
            seed (int): random seed
            fc1_units (int): hidden units in first layer
            fc2_units (int): hidden units in second layer
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)

        # Layers
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

        # (optional) initialize weights
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity="relu")
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, state):
        """Forward pass producing Q-values for each action."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)   # shape: [batch_size, action_size]
