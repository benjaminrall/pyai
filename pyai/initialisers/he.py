from .initialiser import Initialiser
import numpy as np

class HeNormal(Initialiser):
    name = 'he_normal'
    
    def call(self, shape: tuple) -> np.ndarray:
        scale = np.sqrt(2 / shape[-2])
        return np.random.normal(scale=scale, size=shape)
    
class HeUniform(Initialiser):
    name = 'he_uniform'
    
    def call(self, shape: tuple) -> np.ndarray:
        limit = np.sqrt(6 / shape[-2])
        return np.random.uniform(-limit, limit, shape)