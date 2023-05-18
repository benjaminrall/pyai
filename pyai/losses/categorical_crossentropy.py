from .loss import Loss
from pyai.backend.losses import categorical_crossentropy, convert_logits
import numpy as np

class CategoricalCrossentropy(Loss):
    name = "categorical_crossentropy"

    def __init__(self, from_logits: bool = False) -> None:
        self.from_logits = from_logits

    def call(self, output: np.ndarray, target: np.ndarray) -> float:
        return categorical_crossentropy(output, target, self.from_logits)
    
    def derivative(self, output: np.ndarray, target: np.ndarray) -> np.ndarray:
        return convert_logits(output, self.from_logits) - target
    