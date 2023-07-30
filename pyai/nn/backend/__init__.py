"""Neural Networks backend API."""
from pyai.nn.backend.activations import sigmoid
from pyai.nn.backend.activations import stable_sigmoid
from pyai.nn.backend.activations import tanh
from pyai.nn.backend.activations import relu
from pyai.nn.backend.activations import softmax

from pyai.nn.backend.losses import mean_squared_error
from pyai.nn.backend.losses import binary_crossentropy
from pyai.nn.backend.losses import convert_logits
from pyai.nn.backend.losses import categorical_crossentropy

from pyai.nn.backend.regularisers import l1
from pyai.nn.backend.regularisers import l2