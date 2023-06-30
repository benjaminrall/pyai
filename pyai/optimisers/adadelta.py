from pyai.optimisers.optimiser import Optimiser
from pyai.layers.layer import Layer
from pyai.backend.utilities import epsilon
from collections import defaultdict
import numpy as np

class Adadelta(Optimiser):
    name = 'adadelta'

    def __init__(self, eta: float = 1, rho = 0.95) -> None:
        self.eta = eta
        self.rho = rho
        self.one_sub_rho = 1 - rho
        self.grad_avg = defaultdict(lambda : defaultdict(lambda : 0))
        self.delta_avg = defaultdict(lambda : defaultdict(lambda : 0))
        self.epsilon = epsilon()

    def optimise_gradients(self, layer: Layer, gradients: list[np.ndarray]) -> list[np.ndarray]:     
        grad_avg = self.grad_avg[layer]
        delta_avg = self.delta_avg[layer]

        for i in range(len(gradients)):
            grad_avg[i] = self.rho * grad_avg[i] + self.one_sub_rho * np.square(gradients[i])
            gradients[i] = - self.eta * np.sqrt(delta_avg[i] + self.epsilon) / np.sqrt(grad_avg[i] + self.epsilon) * gradients[i]
            delta_avg[i] = self.rho * delta_avg[i] + self.one_sub_rho * np.square(gradients[i])
        return gradients