"""Linear activation function class."""

import numpy as np

from pyai.nn.activations.activation import Activation


class Linear(Activation):
    """Linear activation function (pass-through)."""

    name = "linear"

    def call(self, x: np.ndarray) -> np.ndarray:
        """Applies the linear function to an input."""
        return x

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Applies the derivative of the linear function to an input."""
        return np.full(x.shape, 1)
