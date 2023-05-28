from pyai.optimisers.optimiser import Optimiser
from pyai.optimisers.sgd import SGD 
from pyai.optimisers.rmsprop import RMSprop
from pyai.optimisers.adam import Adam

def get(identifier: str | Optimiser, allow_none: bool = False) -> Optimiser:
    # Returns None in the case that none is allowed
    if identifier is None and allow_none:
        return None
    
    # If identifier is already an instance, then it is simply returned
    if isinstance(identifier, Optimiser):
        return identifier

    # Attempts to get an instance from a dictionary using the string identifier
    return {
        'sgd': SGD(),
        'rmsprop': RMSprop(),
        'adam': Adam()
    }.get(str(identifier).lower(), None if allow_none else SGD())