import numpy as np
import torch


def corr(x, y):
    """
    Compute the correlation between two tensors.
    
    Args:
        x (torch.Tensor): First tensor.
        y (torch.Tensor): Second tensor.
    
    Returns:
        float: The correlation coefficient.
    """
    return torch.corrcoef(torch.stack((x, y)))[0, 1].item()

def normalize_and_clip_rewards(rewards, clip_value=10):
    """
    Normalize and clip rewards.
    
    Args:
        rewards (torch.Tensor): A tensor of rewards of size (M,).
        clip_value (float): The value to clip rewards at. Default is 10.
    
    Returns:
        torch.Tensor: Normalized and clipped rewards.
    """
    # Clip the rewards to be within [-clip_value, +clip_value]
    clipped_rewards = torch.clamp(rewards, -clip_value, clip_value)

    # Normalize the rewards to have zero mean and unit standard deviation
    normalized_rewards = (clipped_rewards - clipped_rewards.mean()) / (clipped_rewards.std() + 1e-8)
    
    return normalized_rewards

def calculate_step_statistics(old_params, new_params):
    abs_max = 0
    mean_square_value = 0
    count = 0

    for old_param, new_param in zip(old_params, new_params):
        param_change = new_param - old_param
        abs_max = max(abs_max, param_change.abs().max().item())
        mean_square_value += (param_change ** 2).sum().item()
        count += param_change.numel()
    
    mean_square_value /= count

    return abs_max, mean_square_value

class RunningStats:
    def __init__(self):
        self.mean = 0
        self.var = 0
        self.count = 0
    
    def update(self, x):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.var += delta * (x - self.mean)
    
    def normalize(self, x):
        std = np.sqrt(self.var / self.count)
        return (x - self.mean) / (std + 1e-8)
    
    def std(self):
        return np.sqrt(self.var / self.count + 1e-8)

class ScaledRewards:
    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.R = 0
        self.running_stats = RunningStats()
    
    def update(self, reward):
        self.R = self.gamma * self.R + reward
        self.running_stats.update(self.R)
        scaled_reward = reward / self.running_stats.std()
        return scaled_reward
    