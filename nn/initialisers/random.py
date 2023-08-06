"""Random initialiser classes."""

import numpy as np

from pyai.nn.initialisers.initialiser import Initialiser


class RandomNormal(Initialiser):
    """Initialiser that generates Numpy arrays with a normal distribution."""

    name = "random_normal"

    def __init__(self, mean: float = 0.0, stddev: float = 0.05) -> None:
        self.mean = mean
        self.stddev = stddev

    def call(self, shape: tuple) -> np.ndarray:
        """Returns a Numpy array of shape `shape` filled with values from a random normal distribution."""
        return np.random.normal(self.mean, self.stddev, shape)

class RandomUniform(Initialiser):
    """Initialiser that generates Numpy arrays with a uniform distribution."""

    name = "random_uniform"

    def __init__(self, low: float = -0.05, high: float = 0.05) -> None:
        self.low = low
        self.high = high

    def call(self, shape: tuple) -> np.ndarray:
        """Returns a Numpy array of shape `shape` filled with values from a random uniform distribution."""
        return np.random.uniform(self.low, self.high, shape)
