import numpy as np

from pyai.backend.utilities import epsilon_clip, normalise_subarrays
from pyai.nn.backend.activations import softmax


def normalise_output(output: np.ndarray, from_logits: bool) -> np.ndarray:
    """Normalises a given output array to be used in a loss function."""
    return softmax(output) if from_logits else normalise_subarrays(output)

def mean_squared_error(output: np.ndarray, target: np.ndarray) -> float:
    """Calculates the mean squared error loss between an output and target."""
    return np.mean(np.square(output - target))

def binary_crossentropy(output: np.ndarray, target: np.ndarray, from_logits: bool = False) -> float:
    """Calculates the binary cross-entropy loss between an output and target."""
    # Use separate stable calculation when using logits
    if from_logits:
        log_result = np.log(1 + np.exp(-np.abs(output)))
        return np.mean(np.maximum(output, 0) - output * target + log_result)

    # Calculates binary cross-entropy from probability distribution
    output = epsilon_clip(normalise_subarrays(output))
    return -np.mean(target * np.log(output) + (1 - target) * np.log(1 - output))

def categorical_crossentropy(output: np.ndarray, target: np.ndarray, from_logits: bool = False) -> float:
    """Calculates the categorical cross-entropy loss between an output and target."""
    output = epsilon_clip(normalise_output(output, from_logits))
    return -np.mean(np.sum(target * np.log(output), axis=-1))
