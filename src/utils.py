import numpy as np


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
    