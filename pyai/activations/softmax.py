from pyai.activations.activation import Activation
from pyai.backend.activations import softmax
from pyai.initialisers import GlorotUniform
import numpy as np

# Softmax activation function
class Softmax(Activation):
    name = 'softmax'

    def call(self, x: np.ndarray) -> np.ndarray:
        return softmax(x)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones(x.shape)