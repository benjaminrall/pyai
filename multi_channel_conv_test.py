import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy.signal import convolve
from scipy.ndimage import grey_dilation

class TestConvLayer():
    # (n_filters, channels, ky, kx)
    filters = np.array([
        [
            [
                [ 1,  2],
                [ 3,  4]
            ],
            [
                [-3,  2],
                [ 4,  1]
            ]
        ],
        [
            [
                [-2,  2],
                [ 1,  5]
            ],
            [   
                [ 7, -1],
                [-4,  3]
            ]
        ],
    ])

    # (filters)
    biases = np.array([-1, 1])

    def __init__(self, filters, kernel_size, strides=(1, 1)) -> None:
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

        self.x = input

        self.z = np.zeros(input.shape[:1] + self.output_shape)
        for i, filter in enumerate(self.filters):
            for j, kernel in enumerate(filter):
                self.z[:, :, :, i] += convolve(input[:, :, :, j], kernel[None], 'valid')
        #self.z = self.z + self.biases

        s0, s1 = input.strides[1:3]
        kernel_strides = (self.strides[0] * s0, self.strides[1] * s1, s0, s1)
        strides_shape = input.strides[:1] + kernel_strides + input.strides[3:]

        self.input_view = as_strided(input, input.shape[:1] + self.view_shape, strides_shape)
        
        self.weight_view = self.input_view.copy()

        return self.z
    
    def backward(self, derivatives: np.ndarray) -> np.ndarray:
        nabla_b = np.sum(derivatives, axis=(0, 1, 2))
        
        # Nabla W -> Each kernel is for specific channel and filter
        nabla_w = np.zeros(self.filters.shape)

        temp_w = np.moveaxis(nabla_w, 1, 3)
        for filter in range(self.filters.shape[0]):
            kernel_deriv = derivatives[:, :, :, filter, None, None, None] * self.input_view[:, :, :, :, :, :]
            deriv_sum = np.sum(kernel_deriv, axis=(0, 1, 2))
            temp_w[filter] = deriv_sum
                
        # Nabla X -> Each channel impacts both filters so must be summed over all dy channels
        weight_view = np.zeros(self.input_view.shape[:-1])
        delta = np.zeros(derivatives.shape[:1] + self.input_shape)

        for filter in range(self.filters.shape[0]):
            for channel in range(derivatives.shape[-1]):
                weight_view[:, :, :] = self.filters[filter, channel]
                scaled_kernel = derivatives[:, :, :, channel, None, None] * weight_view
                rows, cols = scaled_kernel.shape[1:3]
                for row in range(rows):
                    for col in range(cols):
                        my, mx = row * self.strides[0], col * self.strides[1]
                        ny, nx = my + self.kernel_size[0], mx + self.kernel_size[1]
                        delta[:, my:ny, mx:nx, channel] += scaled_kernel[:, row, col]
                
        return delta

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
        [[66, 60], [71, 58], [81, 59]],
        [[33, 67], [55, 9], [77, 27]],
        [[80, 7], [19, 97], [53, 30]]
    ],
    [
        [[66, 60], [71, 58], [81, 59]],
        [[33, 67], [55, 9], [77, 27]],
        [[80, 7], [19, 97], [53, 30]]
    ]
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
print((output.astype(int) == expected_output).all())

delta = layer.backward(derivatives)
print(delta)