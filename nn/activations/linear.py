import numpy as np

from pyai.nn.activations.activation import Activation


class Linear(Activation):
    """Linear activation function (pass-through)."""

    name = "linear"

    def call(self, x: np.ndarray) -> np.ndarray:
        return x

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.full(x.shape, 1)
