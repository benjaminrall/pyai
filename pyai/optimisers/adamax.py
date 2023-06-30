from pyai.optimisers.optimiser import Optimiser
from pyai.layers.layer import Layer
import numpy as np

class Adamax(Optimiser):
    name = 'adamax'

    def optimise_gradients(self, layer: Layer, gradients: list[np.ndarray]) -> list[np.ndarray]:
        raise NotImplementedError()