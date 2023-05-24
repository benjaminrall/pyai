from abc import ABC, abstractmethod
import numpy as np

class Regulariser(ABC):
    name: str

    def __call__(self, x: np.ndarray) -> float:
        return self.call(x)
    
    @abstractmethod
    def call(self, x: np.ndarray) -> float:
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        pass