from collections import defaultdict

import numpy as np

from pyai.backend.utilities import epsilon
from pyai.nn.layers.layer import Layer
from pyai.nn.optimisers.optimiser import Optimiser


class Adamax(Optimiser):
    """Optimiser that implements the Adamax algorithm."""

    name = "adamax"

    def __init__(self, eta: float = 0.001, beta_1: float = 0.9, beta_2: float = 0.999) -> None:
        # Stores parameter values
        self.eta = eta
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.one_sub_beta_1 = 1 - beta_1

        # Initialises 1st moment vector
        self.m = defaultdict(Optimiser.zero_cache)

        # Initialise the exponentially weighted infinity norm
        self.u = defaultdict(Optimiser.zero_cache)

        # Initialise timestep counter
        self.iterations = Optimiser.zero_cache()

        # Stores small constant for numerical stability
        self.epsilon = epsilon()

    def optimise_gradients(self, layer: Layer, gradients: list[np.ndarray]) -> list[np.ndarray]:
        # Increases the layer's iteration counter
        iteration = self.iterations[layer] = self.iterations[layer] + 1

        # Stores local references to the layer's M and U values
        layer_M = self.m[layer]
        layer_U = self.u[layer]

        # Loops through the gradients for each variable in the layer
        for i in range(len(gradients)):
            # Calculates new M and U values
            layer_M[i] = self.beta_1 * layer_M[i] + self.one_sub_beta_1 * gradients[i]
            layer_U[i] = np.maximum(self.beta_2 * layer_U[i], np.abs(gradients[i]))

            # Adjusts learning rate and applies gradient changes
            current_eta = self.eta / (1 - np.power(self.beta_1, iteration))
            gradients[i] = -current_eta * layer_M[i] / (layer_U[i] + self.epsilon)

        return gradients
