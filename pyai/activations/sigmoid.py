from .activation import Activation
from pyai.backend.activations import sigmoid
from pyai.initialisers import GlorotUniform
import numpy as np

# Sigmoid activation function
class Sigmoid(Activation):
    name = "sigmoid"
    weights_initialiser = GlorotUniform()

    def call(self, x: np.ndarray) -> np.ndarray:
        return sigmoid(x)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        s = sigmoid(x)
        return s * (1 - s)