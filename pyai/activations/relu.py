from pyai.activations.activation import Activation
from pyai.backend.activations import relu
from pyai.initialisers import HeNormal
import numpy as np

# ReLU activation function
class ReLU(Activation):
    name = "relu"
    weights_initialiser = HeNormal()

    def call(self, x: np.ndarray) -> np.ndarray:
        return relu(x)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(np.double)