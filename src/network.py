import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCriticNetwork, self).__init__()
        self.policy = nn.Sequential(
            layer_init(nn.Linear(state_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01)
        )
        self.value = nn.Sequential(
            layer_init(nn.Linear(state_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )
    
    def forward(self, state):        
        # Forward pass through policy head to get action probabilities
        action_logits = self.policy(state)
        
        # Forward pass through value head to get state value
        state_value = self.value(state)
        
        return action_logits, state_value.flatten()
    
    def act(self, state):
        # Get action probabilities and state value
        action_logits, state_value = self.forward(state)
        
        # Create a categorical distribution based on action probabilities
        dist = Categorical(logits=action_logits)
        
        # Sample an action from the distribution
        action = dist.sample()
        
        # Return the sampled action, its log probability, and the entropy of the distribution
        return action, dist.log_prob(action), dist.entropy(), state_value
    