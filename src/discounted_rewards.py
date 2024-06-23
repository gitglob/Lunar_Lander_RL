import numpy as np
from .running_stats import RunningStats


class DiscountedRewards:
    """
    A class to compute discounted rewards and maintain running statistics for these discounted rewards.
    """
    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.rewards = []
        self.discounted_rewards = []
        self.running_stats = RunningStats()

    def update(self, reward):
        self.rewards.append(reward)
        self._discount()
        discounted_reward = self.discounted_rewards[-1] 
        normalized_discounted_reward = discounted_reward / (np.sqrt(self.running_stats.var) + 1e-8)
        self.running_stats.update(self.discounted_rewards[-1])

        return normalized_discounted_reward

    def _discount(self):
        """Computes the discounted sum of rewards."""
        running_add = 0
        for reward in reversed(self.rewards):
            running_add = running_add * self.gamma + reward
            self.discounted_rewards.insert(0, running_add)
    
    def reset(self):
        self.rewards.clear()
        self.discounted_rewards.clear()
        self.running_stats.reset()
