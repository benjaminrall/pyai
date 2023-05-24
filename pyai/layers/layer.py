from abc import ABC, abstractmethod
import numpy as np

# Base class for layers in the network 
class Layer(ABC):
    @abstractmethod
    def __init__(self) -> None:
        self.parameters: int = None
        self.input_shape: tuple = None
        self.output_shape: tuple = None
        self.built: bool = False

    def __call__(self, input: np.ndarray) -> np.ndarray:
        return self.forward(input)

    @abstractmethod
    def build(self, input_shape: tuple) -> tuple:
        pass

    @abstractmethod
    def forward(self, input: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, derivatives: np.ndarray, eta: float) -> np.ndarray:
        pass

    def penalty(self) -> float:
        return 0