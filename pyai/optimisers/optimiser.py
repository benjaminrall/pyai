import numpy as np
from collections import defaultdict
from pyai.layers.layer import Layer
from abc import ABC, abstractmethod

class Optimiser(ABC):
    """The class from which all optimisers inherit."""

    name: str

    def __call__(self, layer: Layer, gradients: list) -> None:
        return self.optimise_gradients(layer, gradients)
    
    @staticmethod
    def zero():
        """Returns 0 for use in `zero_dict`."""
        return 0
    
    @staticmethod
    def zero_cache():
        """Returns a default dict that defaults to zeros for optimiser cache."""
        return defaultdict(Optimiser.zero)

    @abstractmethod
    def optimise_gradients(self, layer: Layer, gradients: list[np.ndarray]) -> list[np.ndarray]:
        """Applies the optimisation algorithm to the given gradients."""
