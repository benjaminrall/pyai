from pyai.optimisers.optimiser import Optimiser
import numpy as np

class SGD(Optimiser):
    name = 'sgd'
    
    def __init__(self, eta: float = 0.01, momentum: float = 0) -> None:
        self.eta = eta
        self.momentum = momentum
        self.velocity = None

    def apply_gradients(self, variables: list[np.ndarray], gradients: list[np.ndarray]) -> None:
        if self.velocity is None:
            self.velocity = [0 for _ in range(len(variables))]
        for i in range(len(variables)):
            self.velocity[i] = self.momentum * self.velocity[i] - self.eta * gradients[i]
            variables[i] += self.velocity[i]
