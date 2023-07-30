import numpy as np
from pyai.nn.initialisers.initialiser import Initialiser

class GlorotNormal(Initialiser):
    """The Glorot normal initialiser, also called the Xavier normal initialiser."""

    name = 'glorot_normal'
    
    def call(self, shape: tuple) -> np.ndarray:
        scale = np.sqrt(2 / (shape[-2] + shape[-1]))
        return np.random.normal(scale=scale, size=shape)
    
class GlorotUniform(Initialiser):
    """The Glorot uniform initialiser, also called the Xavier uniform initialiser."""

    name = 'glorot_uniform'
    
    def call(self, shape: tuple) -> np.ndarray:
        limit = np.sqrt(6 / (shape[-2] + shape[-1]))
        return np.random.uniform(-limit, limit, shape)