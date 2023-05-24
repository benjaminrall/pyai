from pyai.regularisers.regulariser import Regulariser
from pyai.backend.regularisers import l2
import numpy as np

class L2(Regulariser):
    name = 'l2'

    def __init__(self, factor: float = 0.01) -> None:
        self.factor = factor
    
    def call(self, x: np.ndarray) -> float:
        return l2(self.factor, x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 2 * self.factor * x