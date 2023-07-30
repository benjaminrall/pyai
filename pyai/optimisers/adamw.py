import numpy as np
from collections import defaultdict
from pyai.optimisers.optimiser import Optimiser
from pyai.layers.layer import Layer
from pyai.backend.utilities import epsilon

class AdamW(Optimiser):
    """Optimiser that implements the AdamW algorithm."""

    name = 'adamw'

    def __init__(self, eta: float = 0.001, weight_decay: float = 0.004, 
                 beta_1: float = 0.9, beta_2: float = 0.999, bias_correction: bool = True) -> None:
        self.eta = eta
        self.weight_decay = weight_decay
        self.beta_1 = beta_1
        self.one_sub_beta_1 = 1 - beta_1
        self.beta_2 = beta_2
        self.one_sub_beta_2 = 1 - beta_2
        self.bias_correction = bias_correction
        self.m = defaultdict(Optimiser.zero_cache)
        self.v = defaultdict(Optimiser.zero_cache)
        self.iterations = Optimiser.zero_cache()
        self.epsilon = epsilon()

    def optimise_gradients(self, layer: Layer, gradients: list[np.ndarray]) -> list[np.ndarray]:
        # Loops through the gradients for each variable in the layer
        iteration = self.iterations[layer] = self.iterations[layer] + 1
        layer_M = self.m[layer]
        layer_V = self.v[layer]
        for i in range(len(gradients)):
            # Calculates the new first and second order moments (M and V)
            corrected_M = layer_M[i] = self.beta_1 * layer_M[i] + self.one_sub_beta_1 * gradients[i]
            corrected_V = layer_V[i] = self.beta_2 * layer_V[i] + self.one_sub_beta_2 * np.square(gradients[i])

            # Calculates the bias-corrected M and V values
            if self.bias_correction:
                corrected_M = layer_M[i] / (1 - np.power(self.beta_1, iteration))
                corrected_V = layer_V[i] / (1 - np.power(self.beta_2, iteration))

            # Applies the adapted learning rate to the gradients with additional weight decay
            gradients[i] = -self.eta * (corrected_M / (np.sqrt(corrected_V) + self.epsilon) 
                                        + self.weight_decay * layer.variables[i])
        
        return gradients