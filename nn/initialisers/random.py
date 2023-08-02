import numpy as np
from pyai.nn.initialisers.initialiser import Initialiser

class RandomNormal(Initialiser):
    """Initialiser that generates tensors with a normal distribution."""

    name = 'random_normal'

    def __init__(self, mean: float = 0.0, stddev: float = 0.05) -> None:
        self.mean = mean
        self.stddev = stddev
    
    def call(self, shape: tuple) -> np.ndarray:
        return np.random.normal(self.mean, self.stddev, shape)
    
class RandomUniform(Initialiser):
    """Initialiser that generates tensors with a uniform distribution."""

    name = 'random_uniform'

    def __init__(self, low: float = -0.05, high: float = 0.05) -> None:
        self.low = low
        self.high = high
    
    def call(self, shape: tuple) -> np.ndarray:
        return np.random.uniform(self.low, self.high, shape)