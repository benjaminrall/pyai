from pyai.activations.activation import Activation
from pyai.activations.linear import Linear
from pyai.activations.sigmoid import Sigmoid
from pyai.activations.tanh import Tanh
from pyai.activations.relu import ReLU
from pyai.activations.softmax import Softmax

def get(name: str) -> Activation:
    return {
        "linear": Linear(),
        "tanh": Tanh(),
        "sigmoid": Sigmoid(),
        "relu": ReLU(),
        "softmax": Softmax()
    }.get(name, Linear())