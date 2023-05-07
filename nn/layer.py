from personallib.maths import Matrix
import random

def create_random_matrix(r, c, min, max):
    m = Matrix(r, c)
    for i in range(r):
        row = []
        for j in range(c):
            row.append(random.uniform(min, max))
        m.set_row(i, row)
    return m

class Layer:
    def __init__(self, size, previousSize = -1):
        self.size = size
        self.values = Matrix(size, 1)
        if previousSize > 0:
            self.weights = create_random_matrix(size, previousSize, -1, 1)
            self.biases = create_random_matrix(size, 1, 0, 1)
        else:
            self.weights = Matrix(0, 0)
            self.biases = Matrix(0, 0)
        self.gradients = [0 for i in range(size)]