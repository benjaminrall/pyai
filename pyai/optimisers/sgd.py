from pyai.optimisers.optimiser import Optimiser
import numpy as np

class SGD(Optimiser):
    name = 'sgd'

    def call(self, output: np.ndarray, target: np.ndarray) -> float:
        return super().call(output, target)
