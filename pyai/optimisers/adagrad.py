from pyai.optimisers.optimiser import Optimiser
from pyai.layers.layer import Layer
from pyai.backend.utilities import epsilon
from collections import defaultdict
import numpy as np

class Adagrad(Optimiser):
    name = 'adagrad'

    def __init__(self, eta: float = 0.01, initial_accumulator_value = 0.1) -> None:
        self.eta = eta
        self.initial_accumulator_value = initial_accumulator_value
        self.accumulators = defaultdict(self.accumulator_cache)
        self.epsilon = epsilon()

    def get_accumulator_value(self):
        return self.initial_accumulator_value

    def accumulator_cache(self):
        return defaultdict(self.get_accumulator_value)

    def optimise_gradients(self, layer: Layer, gradients: list[np.ndarray]) -> list[np.ndarray]:
        layer_accumulator = self.accumulators[layer]

        for i in range(len(gradients)):
            layer_accumulator[i] = layer_accumulator[i] + np.square(gradients[i])
            gradients[i] = -self.eta / np.sqrt(self.epsilon + layer_accumulator[i]) * gradients[i]

        return gradients