import numpy as np

def dilate_matrix(m: np.ndarray, dy, dx) -> np.ndarray:
    if dy <= 1 and dx <= 1:
        return m

    rows, cols = m.shape
    dilated_rows = rows + (rows - 1) * (dy - 1)
    dilated_cols = cols + (cols - 1) * (dx - 1)

    dilated_matrix = np.zeros((dilated_rows, dilated_cols))
    dilated_matrix[::dy, ::dx] = m

    return dilated_matrix

# Example usage:
original_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
dilated_factor = 2

print(dilate_matrix(original_matrix, 3, 2))