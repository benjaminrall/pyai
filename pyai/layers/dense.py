from pyai.layers.layer import Layer
from pyai.optimisers.optimiser import Optimiser
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
        # Sets input and output shapes and counts parameters
        self.input_shape = input_shape
        self.output_shape = (input_shape[:-1]) + (self.units,)
        
        # Calculates trainable parameters for the layer
        self.parameters = self.units * (input_shape[-1] + 1)

        # Initialises weights and biases
        self.weights = self.weight_initialiser((self.input_shape[-1], self.units))
        self.biases = self.bias_initialiser((self.units,))
        
        self.built = True
        return self.output_shape
    

    def forward(self, input: np.ndarray) -> np.ndarray:
        # Builds the layer if it has not yet been built
        if not self.built:
            self.build(input.shape[1:])

        # Stores the input and calculates z output
        self.input = input
        self.z = np.dot(input, self.weights) + self.biases

        # Applies activation function if necessary
        if self.activation is not None:
            return self.activation(self.z)
        return self.z

    def backward(self, derivatives: np.ndarray, optimiser: Optimiser) -> np.ndarray:
        # Calculates derivatives for the activation function if one was applied
        if self.activation is not None:
            derivatives = self.activation.derivative(self.z) * derivatives

        # Calculates derivatives for the layer nodes
        delta = np.dot(derivatives, self.weights.T)

        # Calculates gradients for the weights and biases
        nabla_w = np.dot(self.input.T, derivatives)
        nabla_b = np.sum(derivatives, axis=0)

        # Applies regularisation to the weight gradients
        if self.weight_regulariser is not None:
            nabla_w += self.weight_regulariser.derivative(self.weights)  

        # Optimises gradients
        nabla_w, nabla_b = optimiser(self, [nabla_w, nabla_b])

        # Applies gradients to weights and biases
        self.weights += nabla_w
        self.biases += nabla_b

        return delta
    
    def penalty(self) -> float:
        if self.built and self.weight_regulariser is not None:
            return self.weight_regulariser(self.weights)
        return 0