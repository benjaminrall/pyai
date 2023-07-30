import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy.signal import convolve
from scipy.ndimage import grey_dilation

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

    def __init__(self, filters, kernel_size) -> None:
        self.n_filters = filters
        self.kernel_size = kernel_size
        self.built = False

    def build(self, input_shape: tuple) -> tuple:
        self.input_shape = input_shape

        output_rows = input_shape[0] - self.kernel_size[0] + 1
        output_cols = input_shape[1] - self.kernel_size[1] + 1
        
        self.output_shape = (output_rows, output_cols, self.n_filters)
        self.view_shape = self.output_shape[:2] + self.kernel_size + input_shape[2:]

        self.built = True
        return self.output_shape
    
    def forward(self, input: np.ndarray) -> np.ndarray:
        if not self.built:
            self.build(input.shape[1:])

        self.input = input

        # Creates a view of the input containing each sub-matrix used in the convolution
        self.input_view = as_strided(
            input, input.shape[:1] + self.view_shape, 
            input.strides[:3] + input.strides[1:]
        )

        # Calculates the result of the convolution
        self.z = np.tensordot(self.input_view, self.weights, axes=3)# + self.biases
        return self.z
    
    def backward(self, derivatives: np.ndarray) -> np.ndarray:
        # Calculates nabla b and nabla w
        nabla_b = np.sum(derivatives, axis=(0, 1, 2))
        nabla_w = np.tensordot(self.input_view, derivatives, axes=((0, 1, 2), (0, 1, 2)))

        # Pads the input derivative in order to calculate delta
        py, px = self.kernel_size[0] - 1, self.kernel_size[1] - 1
        pd = np.pad(derivatives, ((0, 0), (py, py), (px, px), (0, 0)))

        # Creates a view of the padded derivative containing the sub-matrices for convolution
        derivative_view_shape = self.input.shape[:3] + self.kernel_size + derivatives.shape[3:]
        derivative_view = as_strided(pd, derivative_view_shape, pd.strides[:3] + pd.strides[1:])

        # Calculates the full convolution of the flipped weights over the input derivatives
        delta = np.tensordot(derivative_view, self.weights[::-1, ::-1], axes=((3, 4, 5), (0, 1, 3)))

        return nabla_b, nabla_w, delta

# (batches, rows, cols, channels)
image = np.array([
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

# (batches, rows, cols, filters)
expected_output = np.array([
    [
        [[30, 89], [56, 51], [84, 41]],
        [[48, 11], [61, 28], [54, 25]],
        [[83, 35], [31, 95], [66, -11]]
    ],
    [
        [[30, 89], [56, 51], [84, 41]],
        [[48, 11], [61, 28], [54, 25]],
        [[83, 35], [31, 95], [66, -11]]
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
expected_nabla_b = np.array([14, 30])
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
        [[-1, 3], [3, 4], [2, 1]],
        [[1, 0], [1, -2], [4, 1]],
        [[-2, 3], [-3, 6], [2, -1]]
    ],
    [
        [[-1, 3], [3, 4], [2, 1]],
        [[1, 0], [1, -2], [4, 1]],
        [[-2, 3], [-3, 6], [2, -1]]
    ],
])



layer = TestConvLayer(2, (2, 2))

output = layer.forward(image)

print(f"Outputs match: {(output.astype(int) == expected_output).all()}")

nabla_b, nabla_w, delta = layer.backward(derivatives)

print(f"Nabla W match: {(expected_nabla_w == nabla_w).all()}")
print(f"Nabla B match: {(expected_nabla_b == nabla_b).all()}")
print(f"Delta match: {(expected_delta == delta).all()}")
