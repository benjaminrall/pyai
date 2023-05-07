import random

class Matrix:
    def __init__(self, rows, columns, matrix = None):
        if matrix is None:
            self.dimensions = (rows, columns)
            self.matrix = [ [ None for c in range (self.dimensions[1]) ] for r in range(self.dimensions[0]) ]
        else:
            self.dimensions = (len(matrix), len([ list(x) for x in zip(*matrix) ]))
            self.matrix = matrix

    def set(self, rowIndex, colIndex, value):
        self.matrix[rowIndex][colIndex] = value

    @staticmethod
    def multiply(value, m):
        for row in range(m.dimensions[0]):
            for col in range(m.dimensions[1]):
                m.set(row, col, value * m.matrix[row][col])
        return m

    @staticmethod
    def multiply_matrices(matrix1, matrix2):
        if matrix1.dimensions[1] != matrix2.dimensions[0]:
            raise Exception(f'Dimension Error: Matrices with dimensions {matrix1.dimensions} and {matrix2.dimensions} cannot be multiplied')
        result = Matrix(matrix1.dimensions[0], matrix2.dimensions[1])
        for row in range(result.dimensions[0]):
            for col in range(result.dimensions[1]):
                result.set(row, col, sum([ n[0] * n[1] for n in zip(matrix1.matrix[row], [ list(x) for x in zip(*matrix2.matrix) ][col]) ]))
        return result

    @staticmethod
    def add(value, m):
        for row in range(m.dimensions[0]):
            for col in range(m.dimensions[1]):
                m.set(row, col, value + m.matrix[row][col])
        return m
    
    @staticmethod
    def add_matrices(matrix1, matrix2):
        if matrix1.dimensions != matrix2.dimensions:
            raise Exception(f'Dimension Error: Matrices with dimensions {matrix1.dimensions} and {matrix2.dimensions} cannot be summed')
        result = Matrix(matrix1.dimensions[0], matrix1.dimensions[1])
        for row in range(result.dimensions[0]):
            for col in range(result.dimensions[1]):
                result.set(row, col, matrix1.matrix[row][col] + matrix2.matrix[row][col])
        return result

    @staticmethod
    def hadamard_product(matrix1, matrix2):
        if matrix1.dimensions != matrix2.dimensions:
            raise Exception(f'Dimension Error: Matrices with dimensions {matrix1.dimensions} and {matrix2.dimensions} cannot be used for the Hadamard product')
        result = Matrix(matrix1.dimensions[0], matrix1.dimensions[1])
        for row in range(result.dimensions[0]):
            for col in range(result.dimensions[1]):
                result.set(row, col, matrix1.matrix[row][col] * matrix2.matrix[row][col])
        return result     

    @staticmethod
    def transpose(m):
        result = Matrix(m.dimensions[1], m.dimensions[0])
        for row in range(m.dimensions[0]):
            for col in range(m.dimensions[1]):
                result.set(col, row, m.matrix[row][col])
        return result

    @staticmethod
    def zero_matrix(rows, columns):
        return Matrix(rows, columns, [ [ 0 for c in range (columns) ] for r in range(rows) ])

    @staticmethod
    def random_matrix(rows, columns, min, max):
        return Matrix(rows, columns, [ [ (random.random() * (max - min)) + min for c in range(columns) ] for r in range(rows) ])