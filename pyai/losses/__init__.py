from pyai.losses.loss import Loss
from pyai.losses.mean_squared_error import MeanSquaredError
from pyai.losses.binary_crossentropy import BinaryCrossentropy
from pyai.losses.categorical_crossentropy import CategoricalCrossentropy

def get(name: str) -> Loss:
    return {
        "mean_squared_error": MeanSquaredError(),
        "binary_crossentropy": BinaryCrossentropy(),
        "categorical_crossentropy": CategoricalCrossentropy()
    }.get(name, MeanSquaredError())