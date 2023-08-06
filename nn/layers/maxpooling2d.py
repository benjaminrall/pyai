import numpy as np

from pyai.nn.layers.layer import Layer


class MaxPooling2D(Layer):
    """A neural network layer that performs the max pooling operation for 2D spatial data."""

    def __init__(self, pool_size: tuple = (2, 2), strides: tuple | None = None) -> None:
        super().__init__()
        self.pool_size = pool_size
        self.strides = strides if strides else pool_size

    def build(self, input_shape: tuple) -> tuple:
        self.input_shape = input_shape

        # Calculates the amount of output rows and columns after pooling
        output_rows = (input_shape[0] - self.pool_size[0]) // self.strides[0] + 1
        output_cols = (input_shape[1] - self.pool_size[1]) // self.strides[1] + 1

        # Stores the output shape and the shape of the view for pooling
        self.output_shape = (output_rows, output_cols) + input_shape[2:]
        self.view_shape = self.output_shape[:2] + self.pool_size + input_shape[2:]

        self.built = True
        return self.output_shape

    def call(self, input: np.ndarray, **kwargs) -> np.ndarray:
        # Builds the layer if it has not yet been built.
        if not self.built:
            self.build(input.shape[1:])

        # Calculates the strides to be used for creating the input view
        s0, s1 = input.strides[1:3]
        pool_strides = (self.strides[0] * s0, self.strides[1] * s1, s0, s1)
        strides_shape = input.strides[:1] + pool_strides + input.strides[3:]

        # Creates a view containing the pools of the input
        pools = np.lib.stride_tricks.as_strided(
            input, input.shape[:1] + self.view_shape, strides=strides_shape
        )

        # Calculates the maxes and the max mask for the backwards pass
        maxes = np.max(pools, axis=(3, 4))
        self.max_mask = pools == maxes[:, :, :, None, None]
        return maxes


    def backward(self, derivatives: np.ndarray, _) -> np.ndarray:
        # Scales the max mask by the derivatives
        derivatives = derivatives[:, :, :, None, None] * self.max_mask

        # Combines the pools back into the input shape to create delta
        rows, cols = derivatives.shape[1:3]
        delta = np.zeros(derivatives.shape[:1] + self.input_shape)
        for row in range(rows):
            for col in range(cols):
                my, mx = row * self.strides[0], col * self.strides[1]
                ny, nx = my + self.pool_size[0], mx + self.pool_size[1]
                delta[:, my:ny, mx:nx] += derivatives[:, row, col]

        return delta
