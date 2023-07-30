import numpy as np
from numpy.lib.stride_tricks import as_strided
from pyai.initialisers import GlorotUniform, Ones, Zeros

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
        [[9, 6], [6, 1], [2, 9], [3, 2]]
    ],

])

M3 = np.array([
    [
        [[3],[8],[7],[7]],
        [[5],[2],[6],[5]],
        [[3],[1],[5],[3]],
        [[9],[6],[2],[3]]
    ],
    [
        [[9],[4],[1],[7]],
        [[1],[3],[2],[6]],
        [[4],[7],[1],[1]],
        [[6],[1],[9],[2]]
    ],
])

M4 = np.array([
    [
        [1, 2, 3], 
        [4, 5, 6], 
        [7, 8, 9]
    ]
])

M5 = np.array([
    [
        [[3, 9], [8, 4], [7, 1], [7, 7]],
        [[5, 1], [2, 3], [6, 2], [5, 6]],
        [[3, 4], [1, 7], [5, 1], [3, 1]],
        [[9, 6], [6, 1], [2, 9], [3, 2]]
    ],
    [
        [[3, 9], [8, 4], [7, 1], [7, 7]],
        [[5, 1], [2, 3], [6, 2], [5, 6]],
        [[3, 4], [1, 7], [5, 1], [3, 1]],
        [[9, 6], [6, 1], [2, 9], [3, 2]]
    ],
])


M = M5

if len(M.shape) == 3:
    M = np.reshape(M, M.shape[:3] + (1,))

num_filters = 2
strides = (1, 1)
kernel_size = (2, 2)

batches, rows, cols = M.shape[:3]
channels = M.shape[3:]

kernel_initialiser = Ones()
bias_initialiser = Zeros()    

output_height = (rows - kernel_size[0]) // strides[0] + 1
output_width = (cols - kernel_size[1]) // strides[1] + 1

output_shape = (batches, output_height, output_width, num_filters)

filters = kernel_initialiser(kernel_size + channels + (num_filters,))
#filters = np.reshape(range(1, 5), filters.shape)
filters = np.array([
    [
        [
            [1, -2],
            [-3, 7],
        ],
        [
            [2, 2],
            [2, -1]
        ]
    ],
    [
        [
            [3, 1],
            [4, -4]
        ],
        [
            [4, 5],
            [1, 3]
        ]
    ]
])

def forward(image: np.ndarray, filters: np.ndarray, strides=(1, 1)):
    Hout = output_shape[1]
    Wout = output_shape[2]

    s0, s1 = image.strides[1:3]

    view_shape = (image.shape[0], Hout, Wout) + kernel_size + channels
    pool_strides = (strides[0] * s0, strides[1] * s1, s0, s1)
    strides_shape = image.strides[:1] + pool_strides + image.strides[3:]

    view = as_strided(image, view_shape, strides_shape)

    return np.tensordot(view, filters, axes=3)

result = forward(M, filters, strides)
print(result)

derivative = np.array([
    [
        [[3], [2]],
        [[1], [0]],
    ]
])

def conv2d(a: np.ndarray, b: np.ndarray, strides):
    Hout = (a.shape[1] - b.shape[0]) // strides[0] + 1
    Wout = (a.shape[2] - b.shape[1]) // strides[1] + 1

    s0, s1 = a.strides[1:3]

    view_shape = (a.shape[0], Hout, Wout) + b.shape[:2] + a.shape[3:]
    pool_strides = (strides[0] * s0, strides[1] * s1, s0, s1)
    strides_shape = a.strides[:1] + pool_strides + a.strides[3:]

    view = np.lib.stride_tricks.as_strided(a, view_shape, strides_shape)
    return np.tensordot(view, b, axes=3)

def backwards(image: np.ndarray, derivatives: np.ndarray, filters: np.ndarray):
    # db = derivatives
    db = np.sum(derivatives, axis=0)

    # dW = image * derivatives
    Hout = image.shape[1] - derivatives.shape[1] + 1
    Wout = image.shape[2] - derivatives.shape[2] + 1

    s0, s1 = image.strides[1:3]
    view_shape = (image.shape[0], Hout, Wout) + kernel_size + image.shape[3:]
    pool_strides = (strides[0] * s0, strides[1] * s1, s0, s1)
    strides_shape = image.strides[:1] + pool_strides + image.strides[3:]

    view = as_strided(image, view_shape, strides_shape)
    print(image)
    print(derivatives)
    print(filters)
    derivatives_view = as_strided(derivatives, derivatives.shape[1:] + derivatives.shape[:1], derivatives.strides[1:] + derivatives.strides[:1])

    dw = np.tensordot(view, derivatives_view, axes=3)
    dw = as_strided(dw, dw.shape[1:] + dw.shape[:1], dw.strides[1:] + dw.strides[:1])
    print(f"dW valid shape: {dw.shape == filters.shape}")

    print(filters, dw)

    # dX = pad(derivatives) * filters
    padded_derivatives = np.pad(derivatives, ((0, 0), (1, 1), (1, 1), (0, 0)))
    dx = conv2d(padded_derivatives, filters[::-1, ::-1], (1, 1))
    print(f"dX valid shape: {dx.shape == image.shape}")

    return dx




backwards(M, result, filters)