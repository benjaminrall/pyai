from pyai.layers.layer import Layer
import pyai.activations as activations
import numpy as np

class Activation(Layer):
    """A neural network layer that applies an activation function to its inputs."""

    def __init__(self, activation: str = ""):
        super().__init__()
        self.activation = activations.get(activation)

    def build(self, input_shape: tuple) -> tuple:
        self.input_shape, self.output_shape = input_shape, input_shape
        return input_shape

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        return self.activation(input) 
    
    def backward(self, derivatives: np.ndarray, _) -> np.ndarray:
        return self.activation.derivative(self.input) * derivatives
