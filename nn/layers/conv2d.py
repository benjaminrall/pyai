"""2D Convolutional layer class."""

import numpy as np
from numpy.lib.stride_tricks import as_strided

import pyai.nn.activations as activations
import pyai.nn.initialisers as initialisers
import pyai.nn.regularisers as regularisers
from pyai.backend import dilate
from pyai.nn.layers.layer import Layer
from pyai.nn.optimisers.optimiser import Optimiser


class Conv2D(Layer):
    """A neural network layer that performs spatial convolution over 2D data."""

    n_variables = 2

    def __init__(self, filters: int,
                 kernel_size: tuple[int, int],
                 strides: tuple[int, int] = (1, 1),
                 activation: str | activations.Activation = None,
                 kernel_initialiser: str | initialisers.Initialiser = "glorot_uniform",
                 bias_initialiser: str | initialisers.Initialiser = "zeros",
                 kernel_regulariser: str | regularisers.Regulariser = None
                 ) -> None:
        super().__init__()
        # Stores filters and kernel size
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
        """Creates and initialises the variables of the 2D Convolutional layer."""
        # Sets input shape
        self.input_shape = input_shape

        # Calculates output rows and cols
        output_rows = (input_shape[0] - self.kernel_size[0]) // self.strides[0] + 1
        output_cols = (input_shape[1] - self.kernel_size[1]) // self.strides[1] + 1

        # Stores the output shape and the shape of the input view for convolution
        self.output_shape = (output_rows, output_cols, self.filters)
        self.view_shape = self.output_shape[:2] + self.kernel_size + input_shape[2:]

        # Initialises kernels and biases
        self.kernels = self.kernel_initialiser((*self.kernel_size, input_shape[-1], self.filters))
        self.biases = self.bias_initialiser((self.filters,))

        self.variables = [self.kernels, self.biases]

        # Calculates trainable parameters for the layer
        self.parameters = np.prod(self.kernels.shape) + np.prod(self.biases.shape)

        self.built = True
        return self.output_shape

    def call(self, input: np.ndarray, **kwargs) -> np.ndarray:
        """Calculates the output of the 2D Convolutional layer for a given input."""
        # Reshapes inputs that don't have channels to have a single channel
        if len(input.shape) == 3:
             input = np.reshape(input, input.shape[:3] + (1,))

        # Builds the layer if it has not yet been built
        if not self.built:
            self.build(input.shape[1:])

        # Stores the current input tensor
        self.input = input

        # Creates a view of the input containing the sub-matrices for convolution
        s0, s1 = input.strides[1:3]
        kernel_strides = (self.strides[0] * s0, self.strides[1] * s1, s0, s1)
        self.input_view = as_strided(
            input, input.shape[:1] + self.view_shape,
            input.strides[:1] + kernel_strides + input.strides[3:]
        )

        # Calculates the valid convolution of the weights over the inputs and adds the biases
        self.z = np.tensordot(self.input_view, self.kernels, axes=3) + self.biases

        # Applies activation function if necessary
        if self.activation is not None:
            return self.activation(self.z)
        return self.z

    def backward(self, derivatives: np.ndarray, optimiser: Optimiser) -> np.ndarray:
        """Performs a backwards pass through the layer and applies gradient updates."""
        # Calculates derivatives for the activation function if one was applied
        if self.activation is not None:
            derivatives = self.activation.derivative(self.z) * derivatives

        # Dilates the matrix to account for varying stride values
        dilated_derivatives = dilate(derivatives, self.strides)

        # Pads the input derivative in order to calculate delta
        py, px = self.kernel_size[0] - 1, self.kernel_size[1] - 1
        pd = np.pad(dilated_derivatives, ((0, 0), (py, py), (px, px), (0, 0)))

        # Creates a view of the padded derivative containing the sub-matrices for convolution
        output_shape = (pd.shape[0], pd.shape[1] - self.kernel_size[0] + 1, pd.shape[2] - self.kernel_size[1] + 1)
        derivative_view_shape = output_shape + self.kernel_size + pd.shape[3:]
        derivative_view = as_strided(pd, derivative_view_shape, pd.strides[:3] + pd.strides[1:])

        # Calculates the full convolution of the flipped weights over the input derivatives
        delta = np.tensordot(derivative_view, self.kernels[::-1, ::-1], axes=((3, 4, 5), (0, 1, 3)))

        # Ensures output is scaled to match the input shape
        if delta.shape != self.input.shape:
            full_delta = np.zeros(self.input.shape)
            full_delta[:, :delta.shape[1], :delta.shape[2]] = delta
            delta = full_delta

        # Calculates gradients for the kernels and biases
        nabla_k = np.tensordot(self.input_view, derivatives, axes=((0, 1, 2), (0, 1, 2)))
        nabla_b = np.sum(derivatives, axis=(0, 1, 2))

        # Applies regularisation to the weight gradients
        if self.kernel_regulariser is not None:
            nabla_k += self.kernel_regulariser.derivative(self.kernels)

        # Optimises gradients
        nabla_k, nabla_b = optimiser(self, [nabla_k, nabla_b])

        # Applies gradients to weights and biases
        self.kernels += nabla_k
        self.biases += nabla_b

        return delta

    def penalty(self) -> float:
        """Calculates the regularisation penalty of the layer."""
        if self.built and self.kernel_regulariser is not None:
            return self.kernel_regulariser(self.kernels)
        return 0

    def set_variables(self, variables: list[np.ndarray]) -> None:
        """Sets the kernels and biases of the layer from a list of numpy arrays."""
        super().set_variables(variables)
        self.kernels = variables[0]
        self.biases = variables[1]
        self.variables = [self.kernels, self.biases]
