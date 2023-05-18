from .activation import Activation
from pyai.backend.activations import tanh
from pyai.initialisers import GlorotUniform
import numpy as np

# Tanh activation function
class Tanh(Activation):
    name = "tanh"
    weights_initialiser = GlorotUniform()

    def call(self, x: np.ndarray) -> np.ndarray:
        return tanh(x)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.square(tanh(x))
