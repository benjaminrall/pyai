"""Hyperbolic tangent activation function class."""

import numpy as np

from pyai.nn.activations.activation import Activation
from pyai.nn.backend.activations import tanh


class Tanh(Activation):
    """Hyperbolic tangent activation function."""

    name = "tanh"

    def call(self, x: np.ndarray) -> np.ndarray:
        """Applies the hyperbolic tangent function to an input."""
        return tanh(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Applies the derivative of the hyperbolic tangent function to an input."""
        return 1 - np.square(tanh(x))
