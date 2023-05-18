from abc import ABC, abstractmethod
import numpy as np

# Base class for initialisers
class Initialiser(ABC):
    name: str

    def __call__(self, shape: tuple) -> np.ndarray:
        return self.call(shape)

    @abstractmethod
    def call(self, shape: tuple) -> np.ndarray:
        pass