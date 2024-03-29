"""L1L2 regulariser class."""

import numpy as np

from pyai.nn.backend.regularisers import l1, l2
from pyai.nn.regularisers.regulariser import Regulariser


class L1L2(Regulariser):
    """A regulariser that applies both L1 and L2 regularisation penalties."""

    name = "l1l2"

    def __init__(self, l1_factor: float = 0.01, l2_factor: float = 0.01) -> None:
        self.l1_factor = l1_factor
        self.l2_factor = l2_factor

    def call(self, x: np.ndarray) -> float:
        """Calculates the L1L2 regularisation penalty."""
        return l1(self.l1_factor, x) + l2(self.l2_factor, x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Calculates the derivative of the L1L2 regularisation penalty."""
        return self.l1_factor * np.sign(x) + 2 * self.l2_factor * x
