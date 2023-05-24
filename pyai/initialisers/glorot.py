from pyai.initialisers.initialiser import Initialiser
import numpy as np

class GlorotNormal(Initialiser):
    name = 'glorot_normal'
    
    def call(self, shape: tuple) -> np.ndarray:
        scale = np.sqrt(2 / (shape[-2] + shape[-1]))
        return np.random.normal(scale=scale, size=shape)
    
class GlorotUniform(Initialiser):
    name = 'glorot_uniform'
    
    def call(self, shape: tuple) -> np.ndarray:
        limit = np.sqrt(6 / (shape[-2] + shape[-1]))
        return np.random.uniform(-limit, limit, shape)