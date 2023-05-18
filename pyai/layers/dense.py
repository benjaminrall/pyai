from .layer import Layer
import pyai.activations as activations
import pyai.initialisers as initialisers
import numpy as np

# A layer that is densely connected to the previous layer through weights and biases
class Dense(Layer):
    def __init__(self, units: int, activation: str = "",
                weight_initialiser: str = "") -> None:
        super().__init__()
        self.units = units
        self.activation = activations.get(activation)
        self.weight_initialiser = initialisers.get(weight_initialiser)
        if self.weight_initialiser is None:
            self.weight_initialiser = self.activation.weights_initialiser

    def build(self, input_shape: tuple) -> tuple:
        # Sets input and output shapes and counts parameters
        self.input_shape = input_shape
        self.output_shape = (input_shape[:-1]) + (self.units,)
        
        self.parameters = self.units * (input_shape[-1] + 1)

        # Initialises weights and biases
        self.weights = self.weight_initialiser((self.input_shape[-1], self.units))
        self.biases = np.zeros(self.units)

        return self.output_shape

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        self.z = np.dot(input, self.weights) + self.biases
        return self.activation(self.z)

    def backward(self, derivatives: np.ndarray, eta: float) -> np.ndarray:
        derivatives = self.activation.derivative(self.z) * derivatives
        
        delta = np.dot(derivatives, self.weights.T)
        nabla_w = np.dot(self.input.T, derivatives)
        nabla_b = np.sum(derivatives, axis=0)

        #lamda = 0.005
        self.weights -= eta * nabla_w# + lamda * self.weights)
        self.biases -= eta * nabla_b

        return delta
