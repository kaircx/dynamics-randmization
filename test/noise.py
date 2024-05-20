import numpy as np

class GaussianNoise:
    def __init__(self, action_dim, mean=0.0, std_dev=0.2):
        self.action_dim = action_dim
        self.mean = mean
        self.std_dev = std_dev

    def sample(self):
        return np.random.normal(self.mean, self.std_dev, self.action_dim)
