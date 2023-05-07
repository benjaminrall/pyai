from abc import ABC, abstractmethod
from weight_initialisers import *
import numpy as np
    
# Base class for all activation functions
class Activation(ABC):
    @abstractmethod
    def __init__(self) -> None:
        self.name: str
        self.weights_initialiser: WeightInitialiser

    @abstractmethod
    def activate(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        pass
    
    @staticmethod
    def get(item: str) -> 'Activation':
        return {
            "linear": Linear(),
            "tanh": Tanh(),
            "sigmoid": Sigmoid(),
            "relu": ReLU(),
        }.get(item, Linear())
    
# Default (Linear) activation function that doesn't change the inputs
class Linear(Activation):
    def __init__(self) -> None:
        super().__init__()
        self.name = "linear"
        self.weights_initialiser = GlorotUniform()

    def activate(self, x: np.ndarray) -> np.ndarray:
        return x
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return x.fill(1)

# Tanh activation function
class Tanh(Activation):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'tanh'
        self.weights_initialiser = GlorotUniform()

    def activate(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        t = self.activate(x)
        return 1 - np.square(t)

# Sigmoid activation function
class Sigmoid(Activation):
    def __init__(self) -> None:
        super().__init__()
        self.name = "sigmoid"
        self.weights_initialiser = GlorotUniform()

    def activate(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        s = self.activate(x)
        return s * (1 - s)
    
# ReLU activation function
class ReLU(Activation):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'relu'
        self.weights_initialiser = HeNormal()

    def activate(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return x > 0