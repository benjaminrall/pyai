from pyai.optimisers.optimiser import Optimiser
from pyai.optimisers.sgd import SGD 
from pyai.optimisers.rmsprop import RMSprop
from pyai.optimisers.adam import Adam
from pyai.optimisers.adamw import AdamW
from pyai.optimisers.adadelta import Adadelta
from pyai.optimisers.adagrad import Adagrad
from pyai.optimisers.adamax import Adamax
from pyai.optimisers.nadam import Nadam

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
        'adadelta': Adadelta(),
        'adagrad': Adagrad(),
        'adamax': Adamax(),
        'adam': Adam(),
        'adamw': AdamW(),
        'nadam': Nadam()
    }.get(str(identifier).lower(), None if allow_none else SGD())