"""Base layer class."""

from abc import ABC, abstractmethod

import numpy as np

import pyai


class Layer(ABC):
    """The class from which all layers inherit."""

    n_variables = 0

    @abstractmethod
    def __init__(self) -> None:
        # Sets default values for shared attributes
        self.input_shape: tuple = None
        self.output_shape: tuple = None
        self.parameters: int = 0
        self.variables: list[np.ndarray] = []
        self.built: bool = False

    def __call__(self, input: np.ndarray, **kwargs) -> np.ndarray:
        """Calculates the output of the layer for a given input."""
        return self.call(input, **kwargs)

    @abstractmethod
    def build(self, input_shape: tuple) -> tuple:
        """Creates and initialises the variables of the layer."""

    @abstractmethod
    def call(self, input: np.ndarray, **kwargs) -> np.ndarray:
        """Calculates the output of the layer for a given input."""

    @abstractmethod
    def backward(self, derivatives: np.ndarray, optimiser: 'pyai.nn.optimisers.Optimiser') -> np.ndarray:
        """Performs a backwards pass through the layer and applies gradient updates if applicable."""

    def penalty(self) -> float:
        """Calculates the regularisation penalty of the layer."""
        return 0

    def get_variables(self) -> list[np.ndarray]:
        """Retrieves the variables of the layer as a list of numpy arrays."""
        if not self.built:
            raise RuntimeError("Cannot get the variables from a layer that hasn't yet been built.")
        return self.variables

    def set_variables(self, variables: list[np.ndarray]) -> None:
        """Sets the variables of the layer from a list of numpy arrays."""
        if not self.built:
            raise RuntimeError("Variables cannot be set for a layer that hasn't yet been built.")
        if len(variables) != self.n_variables or not all([a.shape == b.shape for a, b in zip(self.variables, variables)]):
            raise ValueError("Provided variables do not match the layer's specification.")
