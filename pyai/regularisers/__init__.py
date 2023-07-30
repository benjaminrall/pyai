"""PyAI built-in regularisers."""

from pyai.regularisers.regulariser import Regulariser
from pyai.regularisers.l1 import L1
from pyai.regularisers.l2 import L2
from pyai.regularisers.l1l2 import L1L2

def get(identifier: str | Regulariser, allow_none: bool = False) -> Regulariser:
    """Retrieves a regulariser as a class instance.
    
    The identifier may be the name of a regulariser or a class instance.
    """
    # Returns None in the case that none is allowed
    if identifier is None and allow_none:
        return None
    
    # If identifier is already an instance, then it is simply returned
    if isinstance(identifier, Regulariser):
        return identifier

    # Attempts to get an instance from a dictionary using the string identifier
    return {
        'l1': L1(),
        'l2': L2(),
        'l1l2': L1L2()
    }.get(str(identifier).lower(), None if allow_none else L2())