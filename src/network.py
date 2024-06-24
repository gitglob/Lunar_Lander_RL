import torch.nn as nn
from torch.distributions import Categorical


# Policy (Actor) Network
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        # Define the policy network
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim),  # Output layer
            nn.Softmax(dim=-1)           # Softmax activation for action probabilities
        )
    
    def forward(self, state):
        # Forward pass through the policy network to get action probabilities
        return self.network(state)
    
    def act(self, state):
        # Forward pass through policy network to get action probabilities
        action_probs = self.forward(state)

        # Create a categorical distribution based on action probabilities
        dist = Categorical(action_probs)
        
        # Sample an action from the distribution
        action = dist.sample()

        # Return the sampled action, its log probability, and the entropy of the distribution
        return action, dist.log_prob(action), dist.entropy()

# Value Function (Critic) Network
class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        # Define the value function network
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)            # Output layer for state value
        )
    
    def forward(self, state):    
        return self.network(state)
    