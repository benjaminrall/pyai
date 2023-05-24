from pyai.initialisers.initialiser import Initialiser
import numpy as np

class Zeros(Initialiser):
    name = 'zeros'
    
    def call(self, shape: tuple) -> np.ndarray:
        return np.zeros(shape)
    
class Ones(Initialiser):
    name = 'ones'
    
    def call(self, shape: tuple) -> np.ndarray:
        return np.ones(shape)
    
class Constant(Initialiser):
    name = 'constant'

    def __init__(self, value: float = 0.0) -> None:
        self.value = value
    
    def call(self, shape: tuple) -> np.ndarray:
        return np.full(shape, self.value)