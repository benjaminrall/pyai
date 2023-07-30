import numpy as np
from pyai.initialisers.initialiser import Initialiser

class Zeros(Initialiser):
    """Initialiser that generates tensors initialised to 0."""

    name = 'zeros'
    
    def call(self, shape: tuple) -> np.ndarray:
        return np.zeros(shape)
    
class Ones(Initialiser):
    """Initialiser that generates tensors initialised to 1."""

    name = 'ones'
    
    def call(self, shape: tuple) -> np.ndarray:
        return np.ones(shape)
    
class Constant(Initialiser):
    """Initialiser that generates tensors with constant values."""

    name = 'constant'

    def __init__(self, value: float = 0.0) -> None:
        self.value = value
    
    def call(self, shape: tuple) -> np.ndarray:
        return np.full(shape, self.value)