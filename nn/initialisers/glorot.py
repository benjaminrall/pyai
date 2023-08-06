"""Glorot initialiser classes."""

import numpy as np

from pyai.nn.initialisers.initialiser import Initialiser


class GlorotNormal(Initialiser):
    """The Glorot normal initialiser, also called the Xavier normal initialiser."""

    name = "glorot_normal"

    def call(self, shape: tuple) -> np.ndarray:
        """Returns a Numpy array of shape `shape` filled with values from the Glorot normal initialiser."""
        scale = np.sqrt(2 / (shape[-2] + shape[-1]))
        return np.random.normal(scale=scale, size=shape)

class GlorotUniform(Initialiser):
    """The Glorot uniform initialiser, also called the Xavier uniform initialiser."""

    name = "glorot_uniform"

    def call(self, shape: tuple) -> np.ndarray:
        """Returns a Numpy array of shape `shape` filled with values from the Glorot uniform initialiser."""
        limit = np.sqrt(6 / (shape[-2] + shape[-1]))
        return np.random.uniform(-limit, limit, shape)
