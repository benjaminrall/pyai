"""PyAI built-in initialisers."""

from pyai.initialisers.initialiser import Initialiser
from pyai.initialisers.glorot import GlorotNormal
from pyai.initialisers.glorot import GlorotUniform
from pyai.initialisers.he import HeNormal
from pyai.initialisers.he import HeUniform
from pyai.initialisers.random import RandomNormal
from pyai.initialisers.random import RandomUniform
from pyai.initialisers.constant import Zeros
from pyai.initialisers.constant import Ones
from pyai.initialisers.constant import Constant

def get(identifier: str | Initialiser, allow_none: bool = False) -> Initialiser:
    """Retrieves an initialiser as a class instance.
    
    The identifier may be the name of an initialiser or a class instance.
    """
    # Returns None in the case that none is allowed
    if identifier is None and allow_none:
        return None
    
    # If identifier is already an instance, then it is simply returned
    if isinstance(identifier, Initialiser):
        return identifier

    # Attempts to get an instance from a dictionary using the string identifier
    return {
        'glorot_normal': GlorotNormal(),
        'glorot_uniform': GlorotUniform(),
        'he_normal': HeNormal(),
        'he_uniform': HeUniform(),
        'random_normal': RandomNormal(),
        'random_uniform': RandomUniform(),
        'zeros': Zeros(),
        'ones': Ones(),
        'constant': Constant(),
    }.get(str(identifier).lower(), None if allow_none else GlorotUniform())