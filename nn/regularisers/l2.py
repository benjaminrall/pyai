"""L2 regulariser class."""

import numpy as np

from pyai.nn.backend.regularisers import l2
from pyai.nn.regularisers.regulariser import Regulariser


class L2(Regulariser):
    """A regulariser that applies an L2 regularisation penalty."""

    name = "l2"

    def __init__(self, factor: float = 0.01) -> None:
        self.factor = factor

    def call(self, x: np.ndarray) -> float:
        """Calculates the L2 regularisation penalty."""
        return l2(self.factor, x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Calculates the derivative of the L2 regularisation penalty."""
        return 2 * self.factor * x
