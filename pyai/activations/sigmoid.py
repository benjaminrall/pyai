import numpy as np
from pyai.activations.activation import Activation
from pyai.backend.activations import sigmoid

class Sigmoid(Activation):
    """Sigmoid activation function, `sigmoid(x) = 1 / (1 + exp(-x))`."""

    name = 'sigmoid'

    def call(self, x: np.ndarray) -> np.ndarray:
        return sigmoid(x)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        s = sigmoid(x)
        return s * (1 - s)