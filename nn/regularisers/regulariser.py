from abc import ABC, abstractmethod

import numpy as np


class Regulariser(ABC):
    """The class from which all regularisers inherit."""

    name: str

    def __call__(self, x: np.ndarray) -> float:
        return self.call(x)

    @abstractmethod
    def call(self, x: np.ndarray) -> float:
        """Calculates the regularisation penalty for a given set of inputs."""

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Calculates the derivative of the regulariser for a given set of inputs."""
