import numpy as np

from pyai.nn.backend.regularisers import l2
from pyai.nn.regularisers.regulariser import Regulariser


class L2(Regulariser):
    """A regulariser that applies an L2 regularisation penalty."""

    name = "l2"

    def __init__(self, factor: float = 0.01) -> None:
        self.factor = factor

    def call(self, x: np.ndarray) -> float:
        return l2(self.factor, x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 2 * self.factor * x
