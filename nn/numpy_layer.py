import numpy as np

def create_random_matrix(r, c, min, max):
    return (max - min) * np.random.rand(r, c) + min

def create_empty_matrix(r, c):
    return np.zeros((r, c))

def create_matrix(matrix):
    return np.array(matrix, dtype=np.float64)

class Layer:
    def __init__(self, size, previousSize = -1):
        self.size = size
        self.values = create_empty_matrix(size, 1)
        if previousSize > 0:
            self.weights = create_random_matrix(size, previousSize, -1, 1)
            self.biases = create_random_matrix(size, 1, 0, 1)
        else:
            self.weights = create_empty_matrix(0, 0)
            self.biases = create_empty_matrix(0, 0)
        self.gradients = [ 0 for i in range(size)]

    