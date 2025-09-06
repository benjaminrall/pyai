"""Softmax activation function class."""

import numpy as np
from numpy.typing import NDArray
from pyai.nn.backend import softmax
from .activation import Activation

class Softmax(Activation):
    """
    Softmax activation function.

    Converts vectors of values to probability distributions.
    """

    identifier = 'softmax'

    def call(self, x: NDArray) -> NDArray:
        """Applies the softmax function to an input."""
        return softmax(x)

    def derivative(self, x: NDArray) -> NDArray:
        """Applies the derivative of the softmax function to an input."""
        return np.ones(x.shape)
