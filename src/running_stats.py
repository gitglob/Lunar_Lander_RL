import numpy as np

class RunningStats:
    """
    A class to compute running statistics (mean and variance) for a stream of data.
    This is useful for normalizing data to mean-zero, variance-one vectors, which can help
    stabilize the training of machine learning models.
    
    Attributes:
    -----------
    mean : float
        The running mean of the data.
    var : float
        The running variance of the data.
    count : float
        The count of data points seen so far, initialized to a small value to avoid
        division by zero.
    
    Methods:
    --------
    update(x):
        Updates the running statistics with a new data point.
    normalize(x):
        Normalizes the input data using the running mean and variance.
    """
    
    def __init__(self, epsilon=1e-2):
        """Initializes the RunningStats object."""
        self.mean = 0
        self.var = 1
        self.count = epsilon  # To avoid division by zero

    def update(self, x):
        """Updates the running statistics with a new data point."""
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.var += delta * delta2

    def normalize(self, x):
        """Normalizes the input data using the running mean and variance."""
        return (x - self.mean) / (np.sqrt(self.var / self.count) + 1e-8)
    
    def reset(self):
        self.mean = 0
        self.var = 1

