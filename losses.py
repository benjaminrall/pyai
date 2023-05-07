from abc import ABC, abstractmethod
import numpy as np

class Loss(ABC):
    @abstractmethod
    def __init__(self) -> None:
        self.name: str

    @abstractmethod
    def calculate(self, output: np.ndarray, expected: np.ndarray) -> float:
        pass

    @abstractmethod
    def derivative(self, output: np.ndarray, expected: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def get(item: str) -> 'Loss':
        return {
            "mean_squared_err": MeanSquaredError(),
            "binary_cross_entropy": BinaryCrossEntropy()
        }.get(item, MeanSquaredError())
    
class MeanSquaredError(Loss):
    def __init__(self) -> None:
        super().__init__()
        self.name = "mean_squared_err"

    def calculate(self, output: np.ndarray, expected: np.ndarray) -> float:
        return np.mean(np.square(output - expected))
    
    def derivative(self, output: np.ndarray, expected: np.ndarray) -> np.ndarray:
        return 2 * (output - expected)
    
class BinaryCrossEntropy(Loss):
    def __init__(self) -> None:
        super().__init__()
        self.name = "binary_cross_entropy"

    def calculate(self, output: np.ndarray, expected: np.ndarray) -> float:
        return np.mean(-expected * np.log(output) - (1 - expected) * np.log(1 - output))
    
    def derivative(self, output: np.ndarray, expected: np.ndarray) -> np.ndarray:
        return (expected - output) / (output * (output - 1))