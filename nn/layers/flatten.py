"""Flatten layer class."""

import numpy as np

from pyai.nn.layers.layer import Layer


class Flatten(Layer):
    """A neural network layer that flattens the input."""

    def __init__(self) -> None:
        super().__init__()

    def build(self, input_shape: tuple) -> tuple:
        """Creates and initialises the variables of the Flatten layer."""
        self.input_shape = input_shape
        self.output_shape = (np.prod(input_shape),)
        self.built = True
        return self.output_shape

    def call(self, input: np.ndarray, **kwargs) -> np.ndarray:
        """Calculates the output of the Flatten layer for a given input."""
        # Builds the layer if it has not yet been built.
        if not self.built:
            self.build(input.shape[1:])

        return input.reshape((input.shape[0], -1))

    def backward(self, derivatives: np.ndarray, _) -> np.ndarray:
        """Performs a backwards pass through the layer."""
        return derivatives.reshape(derivatives.shape[:1] + self.input_shape)
