from pyai.layers.layer import Layer
import pyai.activations as activations
import numpy as np

class Dropout(Layer):
    """A neural network layer that applies dropout to the inputs."""

    def __init__(self, rate):
        super().__init__()
        self.rate = rate
        self.inverse_rate = 1 - self.rate
        self.scale = 1 / self.inverse_rate

    def __call__(self, input: np.ndarray, **kwargs) -> np.ndarray:
        if 'training' in kwargs:
            return self.forward(input, kwargs['training'])
        return self.forward(input)

    def build(self, input_shape: tuple) -> tuple:
        self.input_shape, self.output_shape = input_shape, input_shape

        self.built = True
        return input_shape

    def forward(self, input: np.ndarray, training: bool = False) -> np.ndarray:
        # Builds the layer if it has not yet been built.
        if not self.built:
            self.build(input.shape[1:])

        # Returns input unchanged if not currently training
        if not training:
            return input

        # Generates and stores the mask and scale for this pass
        self.mask = np.random.binomial(1, self.inverse_rate, input.shape)
        
        # Applies the mask and scaling factor to the input
        return input * self.mask * self.scale
    
    def backward(self, derivatives: np.ndarray, _) -> np.ndarray:
        # Applies the mask and scaling factor to the derivatives
        return derivatives * self.mask * self.scale
