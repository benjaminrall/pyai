from pyai.optimisers.optimiser import Optimiser
import numpy as np

class Adam(Optimiser):
    name = 'adam'

    def apply_gradients(self, variables: np.ndarray, gradients: np.ndarray) -> None:
        raise NotImplementedError()