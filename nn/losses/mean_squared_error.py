"""Mean Squared Error loss function class."""

import numpy as np

from pyai.nn.backend.losses import mean_squared_error
from pyai.nn.losses.loss import Loss


class MeanSquaredError(Loss):
    """Computes the mean of squares of errors between outputs and targets."""

    name = "mean_squared_error"

    def call(self, output: np.ndarray, target: np.ndarray) -> float:
        """Calculates the result of the mean squared error loss function."""
        return mean_squared_error(output, target)

    def derivative(self, output: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Calculates the derivative of the mean squared error loss function."""
        return 2 * (output - target)
