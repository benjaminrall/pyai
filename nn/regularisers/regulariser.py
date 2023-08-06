"""Base regulariser class."""

from abc import ABC, abstractmethod

import numpy as np


class Regulariser(ABC):
    """The class from which all regularisers inherit."""

    name: str

    def __call__(self, x: np.ndarray) -> float:
        """Calculates the regularisation penalty."""
        return self.call(x)

    @abstractmethod
    def call(self, x: np.ndarray) -> float:
        """Calculates the regularisation penalty."""

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Calculates the derivative of the regulariser."""
