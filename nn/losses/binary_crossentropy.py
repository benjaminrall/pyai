import numpy as np
from pyai.nn.losses.loss import Loss
from pyai.nn.backend.losses import binary_crossentropy

class BinaryCrossentropy(Loss):
    """Computes the cross-entropy loss between the outputs and targets.
    
    This cross-entropy loss is used for binary classification applications
    where the target outputs are either 0 or 1.
    """

    name = "binary_crossentropy"

    def __init__(self, from_logits: bool = False) -> None:
        self.from_logits = from_logits

    def call(self, output: np.ndarray, target: np.ndarray) -> float:
        return binary_crossentropy(output, target, self.from_logits)
    
    def derivative(self, output: np.ndarray, target: np.ndarray) -> np.ndarray:
        return output - target
