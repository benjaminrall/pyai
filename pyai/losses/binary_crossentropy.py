import numpy as np
from pyai.losses.loss import Loss
from pyai.backend.losses import binary_crossentropy

class BinaryCrossentropy(Loss):
    """Computes the binary crossentropy between the outputs and targets."""

    name = 'binary_crossentropy'

    def __init__(self, from_logits: bool = False) -> None:
        self.from_logits = from_logits

    def call(self, output: np.ndarray, target: np.ndarray) -> float:
        return binary_crossentropy(output, target, self.from_logits)
    
    def derivative(self, output: np.ndarray, target: np.ndarray) -> np.ndarray:
        return output - target
    