from pyai.layers.layer import Layer
import pyai.activations as activations
import numpy as np

class Flatten(Layer):
    """A neural network layer that flattens inputs """

    def __init__(self):
        super().__init__()

    def build(self, input_shape: tuple) -> tuple:
        self.input_shape = input_shape
        self.output_shape = (np.prod(input_shape),)
        self.built = True
        return self.output_shape

    def forward(self, input: np.ndarray) -> np.ndarray:
        # Builds the layer if it has not yet been built.
        if not self.built:
            self.build(input.shape[1:])
        return input.reshape((input.shape[0], -1))
    
    def backward(self, derivatives: np.ndarray, _) -> np.ndarray:
        return derivatives.reshape(derivatives.shape[:1] + self.input_shape)