"""Base initialiser class."""

from abc import ABC, abstractmethod
from numpy.typing import NDArray
from pyai.backend import Registrable, Representable

class Initialiser(Representable, Registrable['Initialiser'], ABC):
    """Abstract base class from which all neural network initialisers inherit."""

    identifier: str
    """The initialiser's string identifier."""

    def __call__(self, shape: tuple) -> NDArray:
        """Returns a Numpy array of the given shape filled with values from the initialiser."""
        return self.call(shape)
    
    @abstractmethod
    def call(self, shape: tuple) -> NDArray:
        """Returns a Numpy array of the given shape filled with values from the initialiser."""