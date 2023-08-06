"""He initialiser classes."""

import numpy as np

from pyai.nn.initialisers.initialiser import Initialiser


class HeNormal(Initialiser):
    """He normal initialiser."""

    name = "he_normal"

    def call(self, shape: tuple) -> np.ndarray:
        """Returns a tensor of shape `shape` filled with values from the He normal initialiser."""
        scale = np.sqrt(2 / shape[-2])
        return np.random.normal(scale=scale, size=shape)

class HeUniform(Initialiser):
    """He uniform variance scaling initialiser."""

    name = "he_uniform"

    def call(self, shape: tuple) -> np.ndarray:
        """Returns a tensor of shape `shape` filled with values from the He uniform initialiser."""
        limit = np.sqrt(6 / shape[-2])
        return np.random.uniform(-limit, limit, shape)
