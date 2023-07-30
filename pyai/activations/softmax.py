import numpy as np
from pyai.activations.activation import Activation
from pyai.backend.activations import softmax

class Softmax(Activation):
    """Softmax converts a vector of values to a probability distribution."""

    name = 'softmax'

    def call(self, x: np.ndarray) -> np.ndarray:
        return softmax(x)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones(x.shape)