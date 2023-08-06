from abc import ABC, abstractmethod

import numpy as np


class Loss(ABC):
    """The class from which all loss functions inherit."""

    name: str

    def __call__(self, output: np.ndarray, target: np.ndarray) -> float:
        return self.call(output, target)

    @abstractmethod
    def call(self, output: np.ndarray, target: np.ndarray) -> float:
        """Calculates the result of the loss function for a given output and target."""

    @abstractmethod
    def derivative(self, output: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Calculates the derivative of the loss function for a given output and target."""
