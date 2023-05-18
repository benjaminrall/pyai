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

def get(name: str) -> Initialiser:
    return {
        "glorot_normal": GlorotNormal(),
        "glorot_uniform": GlorotUniform(),
        "he_normal": HeNormal(),
        "he_uniform": HeUniform(),
        "random_normal": RandomNormal(),
        "random_uniform": RandomUniform(),
        "zeros": Zeros(),
        "ones": Ones(),
        "constant": Constant(),
    }.get(name, GlorotNormal())
