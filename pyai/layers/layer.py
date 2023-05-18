from abc import ABC, abstractmethod
import numpy as np

# Base class for layers in the network 
class Layer(ABC):
    @abstractmethod
    def __init__(self) -> None:
        self.input: np.ndarray
        self.parameters: int
        self.input_shape: tuple
        self.output_shape: tuple

    @abstractmethod
    def build(self, input_shape: tuple) -> tuple:
        pass

    @abstractmethod
    def forward(self, input: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, derivatives: np.ndarray, eta: float) -> np.ndarray:
        pass
