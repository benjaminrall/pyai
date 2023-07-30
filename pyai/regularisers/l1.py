import numpy as np
from pyai.regularisers.regulariser import Regulariser
from pyai.backend.regularisers import l1

class L1(Regulariser):
    """A regulariser that applies an L1 regularisation penalty."""

    name = 'l1'

    def __init__(self, factor: float = 0.01) -> None:
        self.factor = factor
    
    def call(self, x: np.ndarray) -> float:
        return l1(self.factor, x)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return self.factor * np.sign(x)