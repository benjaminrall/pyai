from pyai.optimisers.optimiser import Optimiser
from pyai.layers.layer import Layer
import numpy as np

class Adam(Optimiser):
    name = 'adam'

    def optimise_gradients(self, layer: Layer, gradients: list[np.ndarray]) -> list[np.ndarray]:
        raise NotImplementedError()