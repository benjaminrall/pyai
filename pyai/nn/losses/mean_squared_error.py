import numpy as np
from pyai.nn.losses.loss import Loss
from pyai.nn.backend.losses import mean_squared_error

class MeanSquaredError(Loss):
    """Computes the mean of squares of errors between outputs and targets."""

    name = 'mean_squared_error'

    def call(self, output: np.ndarray, target: np.ndarray) -> float:
        return mean_squared_error(output, target)
    
    def derivative(self, output: np.ndarray, target: np.ndarray) -> np.ndarray:
        return 2 * (output - target)
    