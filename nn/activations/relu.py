"""Rectified linear unit activation function class."""

import numpy as np

from pyai.nn.activations.activation import Activation
from pyai.nn.backend.activations import relu


class ReLU(Activation):
    """Rectified linear unit activation function."""

    name = "relu"

    def call(self, x: np.ndarray) -> np.ndarray:
        """Applies the ReLU function to an input."""
        return relu(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Applies the derivative of the ReLU function to an input."""
        return (x > 0).astype(np.double)
