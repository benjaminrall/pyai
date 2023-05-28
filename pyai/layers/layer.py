from abc import ABC, abstractmethod
import numpy as np

class Layer(ABC):
    """Abstract base class for all layers in neural networks."""

    @abstractmethod
    def __init__(self) -> None:
        self.input_shape: tuple = None
        self.output_shape: tuple = None
        self.parameters: int = None
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
    def backward(self, derivatives: np.ndarray, eta: float) -> np.ndarray:
        """Performs a backwards pass through the network and applies parameter changes."""

    def penalty(self) -> float:
        """Returns the regularisation penalty of the layer."""
        return 0
    
    def variables(self) -> tuple[np.ndarray]:
        """Returns the layer's trainable variables."""
        return ()
        
    def gradients(self) -> tuple[np.ndarray]:
        """Returns the layer's trainable variables' gradients."""
        return ()