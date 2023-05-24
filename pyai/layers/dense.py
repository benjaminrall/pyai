from pyai.layers.layer import Layer
import pyai.activations as activations
import pyai.initialisers as initialisers
import pyai.regularisers as regularisers
import numpy as np

class Dense(Layer):
    """Regular densely-connected neural network layer.
    
    `output = activation(dot(input, weights) + bias)`
    """
    
    def __init__(self, units: int, 
                 activation: str | activations.Activation = None,
                 weight_initialiser: str | initialisers.Initialiser = 'glorot_uniform',
                 bias_initialiser: str | initialisers.Initialiser = 'zeros',
                 weight_regulariser: str | regularisers.Regulariser = None
                 ) -> None:
        super().__init__()
        # Ensures that units is an integer
        self.units = int(units) if not isinstance(units, int) else units

        # Gets activation function object
        self.activation = activations.get(activation, True)

        # Gets weight and bias initialiser objects
        self.weight_initialiser = initialisers.get(weight_initialiser)
        self.bias_initialiser = initialisers.get(bias_initialiser)

        # Gets weight rergulariser object
        self.weight_regulariser = regularisers.get(weight_regulariser, True)

    def build(self, input_shape: tuple) -> tuple:
        """Constructs the layer's weight and biases for a given input shape."""
        # Sets input and output shapes and counts parameters
        self.input_shape = input_shape
        self.output_shape = (input_shape[:-1]) + (self.units,)
        
        # Calculates trainable parameters for the layer
        self.parameters = self.units * (input_shape[-1] + 1)

        # Initialises weights and biases
        self.weights = self.weight_initialiser((self.input_shape[-1], self.units))
        self.biases = np.zeros(self.units)

        self.built = True
        return self.output_shape

    def forward(self, input: np.ndarray) -> np.ndarray:
        """Passes an input forward through the layer."""
        if not self.built:
            self.build(input.shape)
        self.input = input
        self.z = np.dot(input, self.weights) + self.biases
        return self.activation(self.z)

    def backward(self, derivatives: np.ndarray, eta: float) -> np.ndarray:
        """Calculates layer derivatives and applies gradients to the parameters."""
        # Calculates z derivatives and uses that to find the layer node derivatives
        z_derivatives = self.activation.derivative(self.z) * derivatives
        derivatives = np.dot(z_derivatives, self.weights.T)

        # Calculates gradients for the weights and biases
        nabla_w = np.dot(self.input.T, z_derivatives)
        nabla_b = np.sum(z_derivatives, axis=0)

        # Applies regularisation to the weight gradients
        if self.weight_regulariser is not None:
            nabla_w += self.input.shape[0] * self.weight_regulariser.derivative(self.weights)

        # Adjusts weights and biases using calculated gradients
        self.weights -= eta * nabla_w  
        self.biases -= eta * nabla_b

        return derivatives
    
    def penalty(self) -> float:
        """Returns the regularisation penalty for the current weights."""
        if self.built and self.weight_regulariser:
            return self.weight_regulariser(self.weights)
        return 0