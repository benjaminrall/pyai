"""PyAI built-in activation functions."""

from pyai.activations.activation import Activation
from pyai.activations.linear import Linear
from pyai.activations.sigmoid import Sigmoid
from pyai.activations.tanh import Tanh
from pyai.activations.relu import ReLU
from pyai.activations.softmax import Softmax

def get(identifier: str | Activation, allow_none: bool = False) -> Activation:
    """Retrieves an activation function as a class instance.
    
    The identifier may be the name of an activation function or a class instance.
    """
    # Returns None in the case that none is allowed
    if identifier is None and allow_none:
        return None
    
    # If identifier is already an instance, then it is simply returned
    if isinstance(identifier, Activation):
        return identifier

    # Attempts to get an instance from a dictionary using the string identifier
    return {
        'linear': Linear(),
        'tanh': Tanh(),
        'sigmoid': Sigmoid(),
        'relu': ReLU(),
        'softmax': Softmax()
    }.get(str(identifier).lower(), None if allow_none else Linear())