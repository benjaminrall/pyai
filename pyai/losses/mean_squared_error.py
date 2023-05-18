from .loss import Loss
from pyai.backend.losses import mean_squared_error
import numpy as np

class MeanSquaredError(Loss):
    name = "mean_squared_error"

    def call(self, output: np.ndarray, target: np.ndarray) -> float:
        return mean_squared_error(output, target)
    
    def derivative(self, output: np.ndarray, target: np.ndarray) -> np.ndarray:
        return 2 * (output - target)
    