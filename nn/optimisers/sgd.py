from collections import defaultdict

import numpy as np

from pyai.nn.layers.layer import Layer
from pyai.nn.optimisers.optimiser import Optimiser


class SGD(Optimiser):
    """Stochastic gradient descent optimiser with momentum."""

    name = "sgd"

    def __init__(self, eta: float = 0.01, momentum: float = 0.0, nesterov: bool = False) -> None:
        self.eta = eta
        self.nesterov = nesterov
        self.momentum = momentum
        self.velocity = defaultdict(Optimiser.zero_cache)

    def optimise_gradients(self, layer: Layer, gradients: list[np.ndarray]) -> list[np.ndarray]:
        # Loops through the gradients for each variable in the layer
        layer_velocity = self.velocity[layer]
        for i in range(len(gradients)):
            # Calculates the gradients scaled by the learning rate
            g = gradients[i] = -self.eta * gradients[i]

            # Applies momentum to the gradients
            if self.momentum > 0:
                v = layer_velocity[i] = self.momentum * layer_velocity[i] + g
                gradients[i] = v if not self.nesterov else self.momentum * v + g

        return gradients
