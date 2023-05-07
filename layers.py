from abc import ABC, abstractmethod
from activations import Activation
from weight_initialisers import WeightInitialiser
import numpy as np

# Base class for layers in the network 
class Layer(ABC):
    @abstractmethod
    def __init__(self) -> None:
        self.input: np.ndarray
        self.parameters: int
        self.input_shape: tuple
        self.output_shape: tuple

    @abstractmethod
    def build(self, input_shape: tuple) -> tuple:
        pass

    @abstractmethod
    def forward(self, input: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, derivatives: np.ndarray, eta: float) -> np.ndarray:
        pass

# A layer which applies an activation function 
class ActivationLayer(Layer):
    def __init__(self, activation: str = ""):
        super().__init__()
        self.activation = Activation.get(activation)

    def build(self, input_shape: tuple) -> tuple:
        self.input_shape, self.output_shape = input_shape, input_shape
        return input_shape

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        return self.activation.activate(input) 
    
    def backward(self, derivatives: np.ndarray, _) -> np.ndarray:
        return self.activation.derivative(self.input) * derivatives


# A layer that is densely connected to the previous layer through weights and biases
class Dense(Layer):
    def __init__(self, units: int, activation: str = "",
                weight_initialiser: str = "") -> None:
        super().__init__()
        self.units = units
        self.activation = Activation.get(activation)
        self.weight_initialiser = WeightInitialiser.get(weight_initialiser)
        if self.weight_initialiser is None:
            self.weight_initialiser = self.activation.weights_initialiser

    def build(self, input_shape: tuple) -> tuple:
        # Sets input and output shapes and counts parameters
        self.input_shape = input_shape
        self.output_shape = (input_shape[:-1]) + (self.units,)
        self.parameters = self.units * (input_shape[-1] + 1)

        # Initialises weights and biases
        self.weights = self.weight_initialiser.initialise(self.input_shape, self.output_shape)
        self.biases = np.zeros(self.units)

        return self.output_shape

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        self.z = np.dot(input, self.weights) + self.biases
        return self.activation.activate(self.z)

    def backward(self, derivatives: np.ndarray, eta: float) -> np.ndarray:
        derivatives = self.activation.derivative(self.z) * derivatives
        
        delta = np.dot(derivatives, self.weights.T)

        nabla_w = np.dot(self.input.T, derivatives)
        nabla_b = np.sum(derivatives, axis=0)

        self.weights -= eta * nabla_w
        self.biases -= eta * nabla_b

        return delta
