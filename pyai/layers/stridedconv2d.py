from pyai.layers.layer import Layer
from pyai.optimisers.optimiser import Optimiser
import pyai.activations as activations
import pyai.initialisers as initialisers
import pyai.regularisers as regularisers
import numpy as np
from scipy.signal import convolve
from numpy.lib.stride_tricks import as_strided

class StridedConv2D(Layer):
    """A neural network layer that performs spatial convolution over 2D data with a strides option."""

    def __init__(self, filters, kernel_size, strides=(1, 1),
                 activation: str | activations.Activation = None,
                 kernel_initialiser: str | initialisers.Initialiser = 'glorot_uniform',
                 bias_initialiser: str | initialisers.Initialiser = 'zeros',
                 kernel_regulariser: str | regularisers.Regulariser = None
                 ):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

        # Gets activation function object
        self.activation = activations.get(activation, True)

        # Gets kernel and bias initialiser objects
        self.kernel_initialiser = initialisers.get(kernel_initialiser)
        self.bias_initialiser = initialisers.get(bias_initialiser)

        # Gets kernel regulariser object
        self.kernel_regulariser = regularisers.get(kernel_regulariser, True)

    def build(self, input_shape: tuple) -> tuple:
        self.input_shape = input_shape

        output_rows = (input_shape[0] - self.kernel_size[0]) // self.strides[0] + 1
        output_cols = (input_shape[1] - self.kernel_size[1]) // self.strides[1] + 1
        
        self.output_shape = (output_rows, output_cols, self.filters)
        self.view_shape = self.output_shape[:2] + self.kernel_size + input_shape[2:]

        self.weights = self.kernel_initialiser((self.filters, input_shape[-1]) + self.kernel_size)
        self.biases = self.bias_initialiser((self.filters,))

        self.built = True
        return self.output_shape

    def call(self, input: np.ndarray, **kwargs) -> np.ndarray:
        # Builds the layer if it has not yet been built
        if len(input.shape) == 3:
             input = np.reshape(input, input.shape[:3] + (1,))

        if not self.built:
            self.build(input.shape[1:])

        self.z = np.zeros(input.shape[:1] + self.output_shape)
        for i, weight in enumerate(self.weights):
            for j, kernel in enumerate(weight):
                self.z[:, :, :, i] += convolve(input[:, :, :, j], kernel[None], 'valid')
        self.z = self.z + self.biases

        s0, s1 = input.strides[1:3]
        kernel_strides = (self.strides[0] * s0, self.strides[1] * s1, s0, s1)
        strides_shape = input.strides[:1] + kernel_strides + input.strides[3:]

        self.input_view = as_strided(input, input.shape[:1] + self.view_shape, strides_shape)
    
        # Applies activation function if necessary
        if self.activation is not None:
            return self.activation(self.z)
        return self.z
            
    def backward(self, derivatives: np.ndarray, optimiser: Optimiser) -> np.ndarray:    
        # Calculates derivatives for the activation function if one was applied
        if self.activation is not None:
            derivatives = self.activation.derivative(self.z) * derivatives

        # Calculates nabla w and nabla b
        nabla_w = np.zeros(self.weights.shape)

        temp_w = np.moveaxis(nabla_w, 1, 3)
        for weight in range(self.weights.shape[0]):
            kernel_deriv = derivatives[:, :, :, weight, None, None, None] * self.input_view[:, :, :, :, :, :]
            deriv_sum = np.sum(kernel_deriv, axis=(0, 1, 2))
            temp_w[weight] = deriv_sum

        nabla_b = np.sum(derivatives, axis=(0, 1, 2))

        # Applies regularisation to the weight gradients
        if self.kernel_regulariser is not None:
            nabla_w += self.kernel_regulariser.derivative(self.weights)

        # Optimises gradients
        nabla_w, nabla_b = optimiser(self, [nabla_w, nabla_b])

        self.weights += nabla_w
        self.biases += nabla_b

        # Nabla X -> Each channel impacts both filters so must be summed over all dy channels
        weight_view = np.zeros(self.input_view.shape[:-1])
        delta = np.zeros(derivatives.shape[:1] + self.input_shape)
        for weight in range(self.filters):
            for channel in range(self.weights.shape[1]):
                weight_view[:, :, :] = self.weights[weight, channel]
                scaled_kernel = derivatives[:, :, :, channel, None, None] * weight_view
                rows, cols = scaled_kernel.shape[1:3]
                for row in range(rows):
                    for col in range(cols):
                        my, mx = row * self.strides[0], col * self.strides[1]
                        ny, nx = my + self.kernel_size[0], mx + self.kernel_size[1]
                        delta[:, my:ny, mx:nx, channel] += scaled_kernel[:, row, col]
                
        return delta
