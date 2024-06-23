import numpy as np

def normalize(self, x):
    return (x - self.mean) / (np.sqrt(self.var) + 1e-8)
