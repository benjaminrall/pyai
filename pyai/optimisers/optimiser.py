from abc import ABC, abstractmethod
from pyai.layers.layer import Layer
import numpy as np

class Optimiser(ABC):
    name: str

    def __call__(self, layer: Layer, gradients: list) -> None:
        return self.optimise_gradients(layer, gradients)

    @abstractmethod
    def optimise_gradients(self, layer: Layer, gradients: list[np.ndarray]) -> list[np.ndarray]:
        """Applies an optimisation algorithm to the given gradients."""
