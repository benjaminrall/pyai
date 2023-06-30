from pyai.optimisers.optimiser import Optimiser
from pyai.layers.layer import Layer
from pyai.backend.utilities import epsilon
from collections import defaultdict
import numpy as np

class Adagrad(Optimiser):
    name = 'adagrad'

    def __init__(self, eta: float = 0.01, initial_accumulator_value = 0.1) -> None:
        self.eta = eta
        self.accumulators = defaultdict(lambda : defaultdict(lambda : initial_accumulator_value))
        self.epsilon = epsilon()

    def optimise_gradients(self, layer: Layer, gradients: list[np.ndarray]) -> list[np.ndarray]:
        layer_accumulator = self.accumulators[layer]

        for i in range(len(gradients)):
            layer_accumulator[i] = layer_accumulator[i] + np.square(gradients[i])
            gradients[i] = -self.eta / np.sqrt(self.epsilon + layer_accumulator[i]) * gradients[i]

        return gradients