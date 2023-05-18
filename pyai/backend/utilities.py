import numpy as np

# Small constant used for numerical stability
_EPSILON = 1e-12

def epsilon() -> float:
    """Returns the small constant used for numerical stability."""
    return _EPSILON

def epsilon_clip(x: np.ndarray) -> np.ndarray:
    """Element-wise value clipping to range [epsilon, 1 - epsilon]."""
    return np.clip(x, _EPSILON, 1 - _EPSILON)

def normalise_subarrays(x: np.ndarray) -> np.ndarray:
    """Normalises subarrays along the last dimension of the input to sum to one."""
    return x / np.sum(x, axis=-1, keepdims=True)

def one_hot_encode(x: np.ndarray, classes: int) -> np.ndarray:
    """Performs one-hot encoding on an input array with a given number of classes."""
    one_hot = np.zeros((x.size, classes))
    one_hot[np.arange(x.size), x] = 1
    return one_hot