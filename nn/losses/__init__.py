"""PyAI built-in loss functions."""

from pyai.nn.losses.loss import Loss
from pyai.nn.losses.mean_squared_error import MeanSquaredError
from pyai.nn.losses.binary_crossentropy import BinaryCrossentropy
from pyai.nn.losses.categorical_crossentropy import CategoricalCrossentropy

def get(identifier: str | Loss, allow_none: bool = False) -> Loss:
    """Retrieves a loss function as a class instance.
    
    The identifier may be the name of a loss function or a class instance.
    """
    # Returns None in the case that none is allowed
    if identifier is None and allow_none:
        return None
    
    # If identifier is already an instance, then it is simply returned
    if isinstance(identifier, Loss):
        return identifier

    # Attempts to get an instance from a dictionary using the string identifier
    return {
        "mean_squared_error": MeanSquaredError(),
        "binary_crossentropy": BinaryCrossentropy(),
        "categorical_crossentropy": CategoricalCrossentropy()
    }.get(str(identifier).lower(), None if allow_none else CategoricalCrossentropy())