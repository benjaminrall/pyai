from pyai.layers.layer import Layer
from pyai.optimisers.optimiser import Optimiser
import pyai.activations as activations
import pyai.initialisers as initialisers
import pyai.regularisers as regularisers
import numpy as np
from scipy.signal import convolve
from numpy.lib.stride_tricks import as_strided

class Conv2D(Layer):
    """A neural network layer that performs spatial convolution over 2D data."""

    n_variables = 2

    def __init__(self, filters: int, kernel_size: tuple[int, int],
                 activation: str | activations.Activation = None,
                 kernel_initialiser: str | initialisers.Initialiser = 'glorot_uniform',
                 bias_initialiser: str | initialisers.Initialiser = 'zeros',
                 kernel_regulariser: str | regularisers.Regulariser = None
                 ):
        super().__init__()
        # Stores filters and kernel size
        self.filters = filters
        self.kernel_size = kernel_size

        # Gets activation function object
        self.activation = activations.get(activation, True)

        # Gets kernel and bias initialiser objects
        self.kernel_initialiser = initialisers.get(kernel_initialiser)
        self.bias_initialiser = initialisers.get(bias_initialiser)

        # Gets kernel regulariser object
        self.kernel_regulariser = regularisers.get(kernel_regulariser, True)

    def build(self, input_shape: tuple) -> tuple:
        # Sets input shape
        self.input_shape = input_shape

        # Calculates output rows and cols
        output_rows = input_shape[0] - self.kernel_size[0] + 1
        output_cols = input_shape[1] - self.kernel_size[1] + 1
        
        # Stores the output shape and the shape of the input view for convolution
        self.output_shape = (output_rows, output_cols, self.filters)
        self.view_shape = self.output_shape[:2] + self.kernel_size + input_shape[2:]

        # Initialises weights and biases
        self.kernel = self.kernel_initialiser(self.kernel_size + (input_shape[-1], self.filters))
        self.biases = self.bias_initialiser((self.filters,))

        self.variables = [self.kernel, self.biases]

        # Calculates trainable parameters for the layer
        self.parameters = np.prod(self.kernel.shape) + np.prod(self.biases.shape)

        self.built = True
        return self.output_shape

    def forward(self, input: np.ndarray) -> np.ndarray:
        # Reshapes inputs that don't have channels to have a single channel
        if len(input.shape) == 3:
             input = np.reshape(input, input.shape[:3] + (1,))

        # Builds the layer if it has not yet been built
        if not self.built:
            self.build(input.shape[1:])

        # Stores the current input tensor
        self.input = input

        # Creates a view of the input containing the sub-matrices for convolution
        self.input_view = as_strided(
            input, input.shape[:1] + self.view_shape, 
            input.strides[:3] + input.strides[1:]
        )

        # Calculates the valid convolution of the weights over the inputs and adds the biases
        self.z = np.tensordot(self.input_view, self.kernel, axes=3) + self.biases

        # Applies activation function if necessary
        if self.activation is not None:
            return self.activation(self.z)
        return self.z
            
    def backward(self, derivatives: np.ndarray, optimiser: Optimiser) -> np.ndarray:    
        # Calculates derivatives for the activation function if one was applied
        if self.activation is not None:
            derivatives = self.activation.derivative(self.z) * derivatives
            
        # Pads the input derivative in order to calculate delta
        py, px = self.kernel_size[0] - 1, self.kernel_size[1] - 1
        pd = np.pad(derivatives, ((0, 0), (py, py), (px, px), (0, 0)))

        # Creates a view of the padded derivative containing the sub-matrices for convolution
        derivative_view_shape = self.input.shape[:3] + self.kernel_size + derivatives.shape[3:]
        derivative_view = as_strided(pd, derivative_view_shape, pd.strides[:3] + pd.strides[1:])

        # Calculates the full convolution of the flipped weights over the input derivatives
        delta = np.tensordot(derivative_view, self.kernel[::-1, ::-1], axes=((3, 4, 5), (0, 1, 3)))
      
        # Calculates gradients for the weights and biases
        nabla_b = np.sum(derivatives, axis=(0, 1, 2))
        nabla_w = np.tensordot(self.input_view, derivatives, axes=((0, 1, 2), (0, 1, 2)))

        # Applies regularisation to the weight gradients
        if self.kernel_regulariser is not None:
            nabla_w += self.kernel_regulariser.derivative(self.kernel)  

        # Optimises gradients
        nabla_w, nabla_b = optimiser(self, [nabla_w, nabla_b])

        # Applies gradients to weights and biases
        self.kernel += nabla_w
        self.biases += nabla_b

        return delta
    
    def penalty(self) -> float:
        if self.built and self.kernel_regulariser is not None:
            return self.kernel_regulariser(self.kernel)
        return 0

    def set_variables(self, variables: list[np.ndarray]):
        super().set_variables(variables)
        self.kernel = variables[0]
        self.biases = variables[1]
        self.variables = [self.kernel, self.biases]