import numpy as np
from collections import defaultdict
from pyai.optimisers.optimiser import Optimiser
from pyai.layers.layer import Layer
from pyai.backend.utilities import epsilon

class Adadelta(Optimiser):
    """Optimiser that implements the Adadelta algorithm."""

    name = 'adadelta'

    def __init__(self, eta: float = 1, rho = 0.95) -> None:
        self.eta = eta
        self.rho = rho
        self.one_sub_rho = 1 - rho
        self.grad_avg = defaultdict(Optimiser.zero_cache)
        self.delta_avg = defaultdict(Optimiser.zero_cache)
        self.epsilon = epsilon()

    def optimise_gradients(self, layer: Layer, gradients: list[np.ndarray]) -> list[np.ndarray]:     
        # Loops through the gradients for each variable in the layer
        grad_avg = self.grad_avg[layer]
        delta_avg = self.delta_avg[layer]
        for i in range(len(gradients)):
            # Maintains a moving discounted average of the square of gradients
            grad_avg[i] = self.rho * grad_avg[i] + self.one_sub_rho * np.square(gradients[i])
            
            # Calculates the optimised gradients using the Adadelta algorithm
            gradients[i] = -self.eta * np.sqrt(delta_avg[i] + self.epsilon) / np.sqrt(grad_avg[i] + self.epsilon) * gradients[i]
            
            # Updates the average delta value for use in the algorithm
            delta_avg[i] = self.rho * delta_avg[i] + self.one_sub_rho * np.square(gradients[i])
        
        return gradients