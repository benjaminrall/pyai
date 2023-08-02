import numpy as np
from pyai.nn.activations.activation import Activation
from pyai.nn.backend.activations import relu

class ReLU(Activation):
    """Applies the rectified linear unit activation function."""

    name = 'relu'

    def call(self, x: np.ndarray) -> np.ndarray:
        return relu(x)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(np.double)