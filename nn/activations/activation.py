import numpy as np
from abc import ABC, abstractmethod

class Activation(ABC):
    """The class from which all activation functions inherit."""

    name: str

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.call(x)

    @abstractmethod
    def call(self, x: np.ndarray) -> np.ndarray:
        """Applies the activation function to an input."""

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Applies the derivative of the activation function to an input."""