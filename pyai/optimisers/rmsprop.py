from pyai.optimisers.optimiser import Optimiser
import numpy as np

class RMSprop(Optimiser):
    name = 'rmsprop'

    def apply_gradients(self, variables: np.ndarray, gradients: np.ndarray) -> None:
        raise NotImplementedError()