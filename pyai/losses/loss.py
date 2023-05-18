from abc import ABC, abstractmethod
import numpy as np

class Loss(ABC):
    name: str

    def __call__(self, output: np.ndarray, target: np.ndarray) -> float:
        return self.call(output, target)

    @abstractmethod
    def call(self, output: np.ndarray, target: np.ndarray) -> float:
        pass

    @abstractmethod
    def derivative(self, output: np.ndarray, target: np.ndarray) -> np.ndarray:
        pass
