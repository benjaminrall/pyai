from abc import ABC, abstractmethod
from pyai.initialisers.initialiser import Initialiser
import numpy as np

# Base class for all activation functions
class Activation(ABC):
    name: str
    weights_initialiser: Initialiser

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.call(x)

    @abstractmethod
    def call(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        pass