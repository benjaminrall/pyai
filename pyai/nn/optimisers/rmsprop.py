import numpy as np
from collections import defaultdict
from pyai.nn.optimisers.optimiser import Optimiser
from pyai.nn.layers.layer import Layer
from pyai.nn.backend.utilities import epsilon

class RMSprop(Optimiser):
    """Optimiser that implements the RMSProp algorithm."""

    name = 'rmsprop'

    def __init__(self, eta: float = 0.001, rho: float = 0.9) -> None:
        self.eta = eta
        self.rho = rho
        self.one_sub_rho = 1 - rho
        self.averages = defaultdict(Optimiser.zero_cache)
        self.epsilon = epsilon()

    def optimise_gradients(self, layer: Layer, gradients: list[np.ndarray]) -> list[np.ndarray]:
        # Loops through the gradients for each variable in the layer
        layer_averages = self.averages[layer]
        for i in range(len(gradients)):
            # Maintains a moving discounted average of the square of gradients
            layer_averages[i] = self.rho * layer_averages[i] + self.one_sub_rho * np.square(gradients[i])
            
            # Divides the gradient by the root of this average
            gradients[i] = -self.eta * gradients[i] / (np.sqrt(layer_averages[i]) + self.epsilon)
        
        return gradients
