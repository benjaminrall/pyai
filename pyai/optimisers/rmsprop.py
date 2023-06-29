from pyai.optimisers.optimiser import Optimiser
from pyai.layers.layer import Layer
from pyai.backend.utilities import epsilon
from collections import defaultdict
import numpy as np

class RMSprop(Optimiser):
    name = 'rmsprop'

    def __init__(self, eta: float = 0.001, rho: float = 0.9) -> None:
        self.eta = eta
        self.rho = rho
        self.one_sub_rho = 1 - rho
        self.averages = defaultdict(lambda : defaultdict(lambda : 0))

    def optimise_gradients(self, layer: Layer, gradients: list[np.ndarray]) -> list[np.ndarray]:
        layer_averages = self.averages[layer]
        for i in range(len(gradients)):
            layer_averages[i] = self.rho * layer_averages[i] + self.one_sub_rho * np.square(gradients[i])
            gradients[i] = -self.eta * gradients[i] / (np.sqrt(layer_averages[i]) + epsilon())
        return gradients