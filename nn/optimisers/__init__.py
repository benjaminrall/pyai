"""PyAI built-in optimisers."""

from pyai.nn.optimisers.adadelta import Adadelta
from pyai.nn.optimisers.adagrad import Adagrad
from pyai.nn.optimisers.adam import Adam
from pyai.nn.optimisers.adamax import Adamax
from pyai.nn.optimisers.adamw import AdamW
from pyai.nn.optimisers.nadam import Nadam
from pyai.nn.optimisers.optimiser import Optimiser
from pyai.nn.optimisers.rmsprop import RMSprop
from pyai.nn.optimisers.sgd import SGD


def get(identifier: str | Optimiser, allow_none: bool = False) -> Optimiser:
    """Retrieves an optimiser as a class instance.

    The identifier may be the name of an optimiser or a class instance.
    """
    # Returns None in the case that none is allowed
    if identifier is None and allow_none:
        return None

    # If identifier is already an instance, then it is simply returned
    if isinstance(identifier, Optimiser):
        return identifier

    # Attempts to get an instance from a dictionary using the string identifier
    return {
        "sgd": SGD(),
        "rmsprop": RMSprop(),
        "adadelta": Adadelta(),
        "adagrad": Adagrad(),
        "adamax": Adamax(),
        "adam": Adam(),
        "adamw": AdamW(),
        "nadam": Nadam()
    }.get(str(identifier).lower(), None if allow_none else RMSprop())
