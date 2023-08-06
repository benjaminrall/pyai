import numpy as np

from pyai.nn.backend.losses import categorical_crossentropy, normalise_output
from pyai.nn.losses.loss import Loss


class CategoricalCrossentropy(Loss):
    """Computes the cross-entropy loss between the outputs and targets."""

    name = "categorical_crossentropy"

    def __init__(self, from_logits: bool = False) -> None:
        self.from_logits = from_logits

    def call(self, output: np.ndarray, target: np.ndarray) -> float:
        return categorical_crossentropy(output, target, self.from_logits)

    def derivative(self, output: np.ndarray, target: np.ndarray) -> np.ndarray:
        return normalise_output(output, self.from_logits) - target
