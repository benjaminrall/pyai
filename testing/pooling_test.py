import numpy as np

M1 = np.array([
    [
        [3,8,7,7],
        [5,2,6,5],
        [3,1,5,3],
        [9,6,2,3]
    ],
    [
        [9,4,1,7],
        [1,3,2,6],
        [4,7,1,1],
        [6,1,9,2]
    ],
])

M2 = np.array([
    [ 
        [[3, 9], [8, 4], [7, 1], [7, 7]],
        [[5, 1], [2, 3], [6, 2], [5, 6]],
        [[3, 4], [1, 7], [5, 1], [3, 1]],
        [[9, 6], [6, 1], [2, 9], [3, 2]],
    ],
    [ 
        [[3, 9], [8, 4], [7, 1], [7, 7]],
        [[5, 1], [2, 3], [6, 2], [5, 6]],
        [[3, 4], [1, 7], [5, 1], [3, 1]],
        [[9, 6], [6, 1], [2, 9], [3, 2]],
    ],
])

M = M2

input_shape = M.shape
strides = (1, 1)
pool_size = (2, 2)

def pool(matrix: np.ndarray):
    s0, s1 = matrix.strides[1:3]
    m1, n1 = matrix.shape[1:3]
    m2, n2 = pool_size
    view_shape = matrix.shape[:1] + (1 + (m1 - m2) // strides[0], 1 + (n1 - n2) // strides[1], m2, n2) + matrix.shape[3:]

    ss = matrix.strides[:1] + (strides[0] * s0, strides[1] * s1, s0, s1) + matrix.strides[3:]
    pools = np.lib.stride_tricks.as_strided(matrix, view_shape, strides=ss)

    return pools

pools = pool(M)

max_pools = np.max(pools, axis=(3, 4))
max_mask = ((pools == max_pools[:, :, :, None, None]))

average_pools = np.mean(pools, axis=(3, 4))
average_mask = np.full(pools.shape, 1 / (pool_size[0] * pool_size[1]))

derivatives = np.array([
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ],
    [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]
    ]
])

derivatives2 = np.array([
    [
        [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]],
        [[0.4, 0.4], [0.5, 0.5], [0.6, 0.6]],
        [[0.7, 0.7], [0.8, 0.8], [0.9, 0.9]]
    ],
    [
        [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]],
        [[0.4, 0.4], [0.5, 0.5], [0.6, 0.6]],
        [[0.7, 0.7], [0.8, 0.8], [0.9, 0.9]]
    ]
])

def backward(derivatives: np.ndarray) -> np.ndarray:
    derivatives = derivatives[:, :, :, None, None] * max_mask

    rows, cols = derivatives.shape[1:3]

    delta = np.zeros(input_shape)

    for row in range(rows):
        for col in range(cols):
            Ty, Tx = row * strides[0], col * strides[1]
            Ey, Ex = Ty + pool_size[0], Tx + pool_size[1]
            delta[:, Ty:Ey, Tx:Ex] += derivatives[:, row, col]
    
    return delta

print(M)
print(backward(derivatives2))