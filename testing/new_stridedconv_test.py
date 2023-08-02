import numpy as np
from numpy.lib.stride_tricks import as_strided

class TestConvLayer():
    # (ky, kx, channels, filters)
    weights = np.array([
        [
            [[1, -2], [-3, 7]],
            [[2, 2], [2, -1]]
        ],
        [
            [[3, 1], [4, -4]],
            [[4, 5], [1, 3]]
        ]
    ])

    # (n_filters,)
    biases = np.array([-1, 1])

    def __init__(self, filters, kernel_size, strides=(1,1)) -> None:
        self.n_filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.built = False

    def build(self, input_shape: tuple) -> tuple:
        self.input_shape = input_shape

        output_rows = (input_shape[0] - self.kernel_size[0]) // self.strides[0] + 1
        output_cols = (input_shape[1] - self.kernel_size[1]) // self.strides[1] + 1
        
        self.output_shape = (output_rows, output_cols, self.n_filters)
        self.view_shape = self.output_shape[:2] + self.kernel_size + input_shape[2:]

        self.built = True
        return self.output_shape
    
    def forward(self, input: np.ndarray) -> np.ndarray:
        if not self.built:
            self.build(input.shape[1:])

        self.input = input

        # Creates a view of the input containing the sub-matrices for convolution
        s0, s1 = input.strides[1:3]
        kernel_strides = (self.strides[0] * s0, self.strides[1] * s1, s0, s1)
        self.input_view = as_strided(
            input, input.shape[:1] + self.view_shape, 
            input.strides[:1] + kernel_strides + input.strides[3:]
        )

        # Calculates the result of the convolution
        self.z = np.tensordot(self.input_view, self.weights, axes=3)# + self.biases
        return self.z
    
    def backward(self, derivatives: np.ndarray) -> np.ndarray:
        # Calculates nabla b and nabla w
        nabla_b = np.sum(derivatives, axis=(0, 1, 2))
        nabla_w = np.tensordot(self.input_view, derivatives, axes=((0, 1, 2), (0, 1, 2)))

        # Pads the input derivative in order to calculate delta
        dilated_derivatives = dilate_matrix(derivatives, self.strides)
        py, px = self.kernel_size[0] - 1, self.kernel_size[1] - 1
        pd = np.pad(dilated_derivatives, ((0, 0), (py, py), (px, px), (0, 0)))
        
        # Creates a view of the padded derivative containing the sub-matrices for convolution
        output_size = (pd.shape[1] - self.kernel_size[0] + 1, pd.shape[2] - self.kernel_size[1] + 1)
        derivative_view_shape = self.input.shape[:1] + output_size + self.kernel_size + pd.shape[3:]
        derivative_view = as_strided(pd, derivative_view_shape, pd.strides[:3] + pd.strides[1:])

        # Calculates the full convolution of the flipped weights over the input derivatives
        delta = np.tensordot(derivative_view, self.weights[::-1, ::-1], axes=((3, 4, 5), (0, 1, 3)))

        #
        if delta.shape != self.input.shape:
            full_delta = np.zeros(self.input.shape)
            full_delta[:, :delta.shape[1], :delta.shape[2]] = delta
            delta = full_delta

        return nabla_b, nabla_w, delta

def dilate_matrix(m: np.ndarray, dilation_factor: tuple) -> np.ndarray:
    if dilation_factor == (1, 1):
        return m
    
    dy, dx = dilation_factor

    batches, rows, cols, channels = m.shape
    dilated_rows = rows + (rows - 1) * (dy - 1)
    dilated_cols = cols + (cols - 1) * (dx - 1)

    dilated_matrix = np.zeros((batches, dilated_rows, dilated_cols, channels))
    dilated_matrix[:, ::dy, ::dx] = m

    return dilated_matrix

# (batches, rows, cols, channels)
image = np.array([
    [
        [[3, 9], [8, 4], [7, 1], [7, 7], [2, 1]],
        [[5, 1], [2, 3], [6, 2], [5, 6], [1, 8]],
        [[3, 4], [1, 7], [5, 1], [3, 1], [9, 9]],
        [[9, 6], [6, 1], [2, 9], [3, 2], [8, 4]],
        [[3, 7], [7, 5], [1, 6], [5, 3], [4, 6]]
    ],
    [
        [[3, 9], [8, 4], [7, 1], [7, 7], [2, 1]],
        [[5, 1], [2, 3], [6, 2], [5, 6], [1, 8]],
        [[3, 4], [1, 7], [5, 1], [3, 1], [9, 9]],
        [[9, 6], [6, 1], [2, 9], [3, 2], [8, 4]],
        [[3, 7], [7, 5], [1, 6], [5, 3], [4, 6]]
    ],
])

# (batches, rows, cols, filters)
expected_output = np.array([
    [
        [[30, 89], [84, 41]],
        [[83, 35], [66, -11]]
    ],
    [
        [[30, 89], [84, 41]],
        [[83, 35], [66, -11]]
    ],
])

# (ky, kx, channels, n_filters)
expected_nabla_w = np.array([
    [
        [[134, 120], [-20, 186]],
        [[88, 164], [54, 102]]
    ],
    [
        [[10, 186], [60, 38]],
        [[64, 110], [4, 154]]
    ]
])

# (n_filters,)
expected_nabla_b = np.array([2, 12])

# (batches, rows, cols, channels)
expected_delta = np.array([
    [
        [[-7, 24], [-1, 14], [14, 3], [6, 3]],
        [[1, -19], [31, -11], [39, 18], [23, 12]],
        [[-5, 31], [-8, 57], [17, -18], [23, 12]],
        [[-3, -20], [4, -29], [23, 27], [3, -1]]
    ],
    [
        [[-7, 24], [-1, 14], [14, 3], [6, 3]],
        [[1, -19], [31, -11], [39, 18], [23, 12]],
        [[-5, 31], [-8, 57], [17, -18], [23, 12]],
        [[-3, -20], [4, -29], [23, 27], [3, -1]]
    ],
])

# (batches, rows, cols, channels)
derivatives = np.array([
    [
        [[-1, 3], [2, 1]],
        [[-2, 3], [2, -1]]
    ],
    [
        [[-1, 3], [2, 1]],
        [[-2, 3], [2, -1]]
    ],
])



layer = TestConvLayer(2, (2, 2), (2, 2))

output = layer.forward(image)


print(f"Outputs match: {(output.astype(int) == expected_output).all()}")

import time
s = time.perf_counter()
for i in range(10000):
    nabla_b, nabla_w, delta = layer.backward(derivatives)
print(f"{time.perf_counter() - s}")

print(delta)

print(f"Nabla W match: {(expected_nabla_w == nabla_w).all()}")
print(f"Nabla B match: {(expected_nabla_b == nabla_b).all()}")
#print(f"Delta match: {(expected_delta == delta).all()}")
