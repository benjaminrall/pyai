from abc import ABC, abstractmethod
import numpy as np
import math

# Base class for Dense layer weight initialisers
class WeightInitialiser(ABC):
    @abstractmethod
    def __init__(self) -> None:
        self.name: str

    @abstractmethod
    def initialise(self, input_shape: tuple, output_shape: tuple) -> np.ndarray:
        pass

    @staticmethod
    def get(item: str) -> 'WeightInitialiser':
        return {
            "glorot_uniform": GlorotUniform(),
            "he_normal": HeNormal()
        }.get(item, GlorotUniform())
    
# Normalised Xavier/Glorot weight initialisation using a Uniform Distribution
class GlorotUniform(WeightInitialiser):
    def __init__(self) -> None:
        super().__init__()
        self.name = "glorot_uniform"

    def initialise(self, input_shape: tuple, output_shape: tuple) -> np.ndarray:
        weights_shape = (input_shape[-1], output_shape[-1])
        limit = math.sqrt(6) / math.sqrt(weights_shape[0] + weights_shape[1])
        return np.random.uniform(-limit, limit, weights_shape)
    
# He weight initialisation using a Gaussian Distrubution 
class HeNormal(WeightInitialiser):
    def __init__(self) -> None:
        super().__init__()
        self.name = "he_normal"

    def initialise(self, input_shape: tuple, output_shape: tuple) -> np.ndarray:
        weights_shape = (input_shape[-1], output_shape[-1])
        deviation = math.sqrt(2 / input_shape[-1])
        return np.random.normal(0, deviation, weights_shape)