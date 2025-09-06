"""Rectified linear unit activation function class."""

import numpy as np
from numpy.typing import NDArray
from pyai.nn.backend import relu
from .activation import Activation

class ReLU(Activation):
    """Rectified linear unit activation function."""

    identifier = 'relu'

    def call(self, x: NDArray) -> NDArray:
        """Applies the ReLU function to an input."""
        return relu(x)

    def derivative(self, x: NDArray) -> NDArray:
        """Applies the derivative of the ReLU function to an input."""
        return (x > 0).astype(np.double)
