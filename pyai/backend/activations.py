import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Applies the sigmoid activation function to the input array."""
    return 1. / (1. + np.exp(-x))

def stable_sigmoid(x: np.ndarray) -> np.ndarray:
    """Applies a numerically stable sigmoid activation function to the input array."""
    # Calculates exponentials for negative and positive values separately
    neg_exp = np.exp(x[x < 0])
    pos_exp = np.exp[-x[x >= 0]]

    # Creates output array using separate equations for stability
    z = np.zeros(x.shape)
    z[x < 0] = neg_exp / (1. + neg_exp)
    z[x >= 0] = 1. / (1. + pos_exp)
    return z

def tanh(x: np.ndarray) -> np.ndarray:
    """Applies the tanh activation function to the input array."""
    return np.tanh(x)

def relu(x: np.ndarray) -> np.ndarray:
    """Applies the Rectified Linear Unit (ReLU) activation function to the input array."""
    return np.maximum(x, 0)

def softmax(x: np.ndarray) -> np.ndarray:
    """Applies the softmax function to the input array."""
    # Ensure numerical stability by subtracting the maximum value from the input
    max_val = np.max(x, axis=-1, keepdims=True)

    # Calculate exponentials and their sums
    exps = np.exp(x - max_val)
    sums = np.sum(exps, axis=-1, keepdims=True)

    # Return final probabilities array
    return exps / sums