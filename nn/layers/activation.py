import numpy as np

import pyai.nn.activations as activations
from pyai.nn.layers.layer import Layer


class Activation(Layer):
    """A neural network layer that applies an activation function to its inputs."""

    def __init__(self, activation: str | activations.Activation) -> None:
        super().__init__()
        self.activation = activations.get(activation)

    def build(self, input_shape: tuple) -> tuple:
        self.input_shape, self.output_shape = input_shape, input_shape

        self.built = True
        return input_shape

    def call(self, input: np.ndarray, **kwargs) -> np.ndarray:
        # Builds the layer if it has not yet been built.
        if not self.built:
            self.build(input.shape[1:])

        # Returns the input with the activation function applied to it
        self.input = input
        return self.activation(input)

    def backward(self, derivatives: np.ndarray, _) -> np.ndarray:
        return self.activation.derivative(self.input) * derivatives
