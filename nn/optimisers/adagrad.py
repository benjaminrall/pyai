"""Adagrad optimiser class."""

from collections import defaultdict

import numpy as np

from pyai.backend.utilities import epsilon
from pyai.nn.layers.layer import Layer
from pyai.nn.optimisers.optimiser import Optimiser


class Adagrad(Optimiser):
    """Optimiser that implements the Adagrad algorithm."""

    name = "adagrad"

    def __init__(self, eta: float = 0.01, initial_accumulator_value = 0.1) -> None:
        self.eta = eta
        self.initial_accumulator_value = initial_accumulator_value
        self.accumulators = defaultdict(self.accumulator_cache)
        self.epsilon = epsilon()

    def get_accumulator_value(self):
        """Gets the initial accumulator value."""
        return self.initial_accumulator_value

    def accumulator_cache(self):
        """Returns a default dict for the accumulator cache."""
        return defaultdict(self.get_accumulator_value)

    def optimise_gradients(self, layer: Layer, gradients: list[np.ndarray]) -> list[np.ndarray]:
        """Applies the Adagrad optimisation algorithm to the given gradients."""
        # Loops through the gradients for each variable in the layer
        layer_accumulator = self.accumulators[layer]
        for i in range(len(gradients)):
            # Adds the square of the current gradients to the accumulator
            layer_accumulator[i] = layer_accumulator[i] + np.square(gradients[i])

            # Uses the accumulator value to adjust the learning rate for each parameter
            gradients[i] = -self.eta / np.sqrt(self.epsilon + layer_accumulator[i]) * gradients[i]

        return gradients
