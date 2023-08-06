"""Softmax activation function class."""

import numpy as np

from pyai.nn.activations.activation import Activation
from pyai.nn.backend.activations import softmax


class Softmax(Activation):
    """Softmax activation function.

    Converts all input vectors to probability distributions.
    """

    name = "softmax"

    def call(self, x: np.ndarray) -> np.ndarray:
        """Applies the softmax function to an input."""
        return softmax(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Applies the derivative of the softmax function to an input."""
        return np.ones(x.shape)
