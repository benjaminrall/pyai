"""Leaky version of the rectified linear unit activation function class."""

import numpy as np

from pyai.nn.activations.activation import Activation
from pyai.nn.backend.activations import leaky_relu


class LeakyReLU(Activation):
    """Rectified linear unit activation function."""

    name = "leaky_relu"

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def call(self, x: np.ndarray) -> np.ndarray:
        """Applies the Leaky ReLU function to an input."""
        return leaky_relu(x, self.alpha)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Applies the derivative of the Leaky ReLU function to an input."""
        return np.where(x >= 0, 1, self.alpha).astype(np.double)
