"""L1 regulariser class."""

import numpy as np

from pyai.nn.backend.regularisers import l1
from pyai.nn.regularisers.regulariser import Regulariser


class L1(Regulariser):
    """A regulariser that applies an L1 regularisation penalty."""

    name = "l1"

    def __init__(self, factor: float = 0.01) -> None:
        self.factor = factor

    def call(self, x: np.ndarray) -> float:
        """Calculates the L1 regularisation penalty."""
        return l1(self.factor, x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Calculates the derivative of the L1 regularisation penalty."""
        return self.factor * np.sign(x)
