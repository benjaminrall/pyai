from abc import ABC, abstractmethod
import numpy as np

class Optimiser(ABC):
    name: str

    def __call__(self, variables: np.ndarray, gradients: np.ndarray) -> None:
        self.apply_gradients(variables, gradients)

    @abstractmethod
    def apply_gradients(self, variables: np.ndarray, gradients: np.ndarray) -> None:
        """Applies gradients to the given variables."""
