"""Base activation function class."""

from abc import ABC, abstractmethod

import numpy as np


class Activation(ABC):
    """The class from which all activation functions inherit."""

    name: str

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Applies the activation function to an input."""
        return self.call(x)

    @abstractmethod
    def call(self, x: np.ndarray) -> np.ndarray:
        """Applies the activation function to an input."""

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Applies the derivative of the activation function to an input."""
