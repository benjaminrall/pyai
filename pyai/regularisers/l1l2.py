from pyai.regularisers.regulariser import Regulariser
from pyai.backend.regularisers import l1, l2
import numpy as np

class L1L2(Regulariser):
    name = 'l1l2'

    def __init__(self, l1_factor: float = 0.01, l2_factor: float = 0.01) -> None:
        self.l1_factor = l1_factor
        self.l2_factor = l2_factor
    
    def call(self, x: np.ndarray) -> float:
        return l1(self.l1_factor, x) + l2(self.l2_factor, x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return self.l1_factor * np.sign(x) + 2 * self.l2_factor * x