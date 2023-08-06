"""Sigmoid activation function class."""

import numpy as np

from pyai.nn.activations.activation import Activation
from pyai.nn.backend.activations import sigmoid


class Sigmoid(Activation):
    """Sigmoid activation function, `sigmoid(x) = 1 / (1 + exp(-x))`."""

    name = "sigmoid"

    def call(self, x: np.ndarray) -> np.ndarray:
        """Applies the sigmoid function to an input."""
        return sigmoid(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Applies the derivative of the sigmoid function to an input."""
        s = sigmoid(x)
        return s * (1 - s)
