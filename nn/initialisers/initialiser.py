from abc import ABC, abstractmethod

import numpy as np


class Initialiser(ABC):
    """The class from which all initialisers inherit."""

    name: str

    def __call__(self, shape: tuple) -> np.ndarray:
        return self.call(shape)

    @abstractmethod
    def call(self, shape: tuple) -> np.ndarray:
        """Returns a tensor of shape `shape` filled with values from the initialiser."""
