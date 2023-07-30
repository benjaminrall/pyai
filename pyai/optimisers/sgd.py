from pyai.optimisers.optimiser import Optimiser
from pyai.layers.layer import Layer
import numpy as np
from collections import defaultdict

class SGD(Optimiser):
    name = 'sgd'
    
    def __init__(self, eta: float = 0.01, momentum: float = 0, nesterov: bool = False) -> None:
        self.eta = eta
        self.nesterov = nesterov
        self.momentum = momentum
        self.velocity = defaultdict(Optimiser.zero_cache)

    def optimise_gradients(self, layer: Layer, gradients: list[np.ndarray]) -> list[np.ndarray]:
        for i in range(len(gradients)):
            # Calculates the gradients scaled by the learning rate
            g = gradients[i] = -self.eta * gradients[i]
            
            # Applies momentum to the gradients
            if self.momentum > 0:
                v = self.velocity[layer][i] = self.momentum * self.velocity[layer][i] + g
                gradients[i] = v if not self.nesterov else self.momentum * v + g
        return gradients