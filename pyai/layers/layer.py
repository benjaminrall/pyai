from abc import ABC, abstractmethod
import numpy as np

class Layer(ABC):
    """Abstract base class for all layers in neural networks."""

    layers = 0

    @abstractmethod
    def __init__(self) -> None:
        # Sets default values for shared attributes
        self.input_shape: tuple = None
        self.output_shape: tuple = None
        self.parameters: int = None
        self.variables: list[np.ndarray] = []
        self.built: bool = False

    def __call__(self, input: np.ndarray) -> np.ndarray:
        return self.forward(input)

    @abstractmethod
    def build(self, input_shape: tuple) -> tuple:
        """Initialises the layer's variables."""

    @abstractmethod
    def forward(self, input: np.ndarray) -> np.ndarray:
        """Passes an input forward through the layer."""

    @abstractmethod
    def backward(self, derivatives: np.ndarray, optimiser) -> tuple[np.ndarray, np.ndarray]:
        """Performs a backwards pass through the network and applies parameter changes."""

    def penalty(self) -> float:
        """Returns the regularisation penalty of the layer."""
        return 0