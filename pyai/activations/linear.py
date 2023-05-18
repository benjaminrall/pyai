from .activation import Activation
from pyai.initialisers import GlorotUniform
import numpy as np

# Default (Linear) activation function that doesn't change the inputs
class Linear(Activation):
    name = "linear"
    weights_initialiser = GlorotUniform()

    def call(self, x: np.ndarray) -> np.ndarray:
        return x
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.full(x.shape, 1)
