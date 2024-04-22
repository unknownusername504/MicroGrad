import numpy as np

from micrograd.tensors.tensor import Function


class MaxPool2D(Function):
    def __init__(self, inputs, output, stride, padding):
        super().__init__(inputs, output)
        self.stride = stride
        self.padding = padding
        # TODO: Preallocate the output tensor

    def _forward(self):
        self.output.value = self.max_pool2d(
            self.inputs[0].value, self.stride, self.padding
        )

    def _backward(self):
        if self.inputs[0].requires_grad:
            self.inputs[0].grad = self.inputs[0].grad + self.max_pool2d_grad(
                self.inputs[0].value,
                self.output.value,
                self.output.grad,
                self.stride,
                self.padding,
            )

    def max_pool2d(self, x, stride, padding):
        # Get the dimensions of the input
        n, c, h, w = x.shape
        # Calculate the output dimensions
        oh = (h + 2 * padding - 1) // stride + 1
        ow = (w + 2 * padding - 1) // stride + 1
        # Pad the input
        x = np.pad(
            x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), "constant"
        )
        # Create the output
        y = np.zeros((n, c, oh, ow))
        # Perform the max pooling
        for i in range(oh):
            for j in range(ow):
                y[:, :, i, j] = np.max(
                    x[
                        :,
                        :,
                        i * stride : i * stride + 2 * padding + 1,
                        j * stride : j * stride + 2 * padding + 1,
                    ],
                    axis=(2, 3),
                )
        return y

    def max_pool2d_grad(self, x, y, dy, stride, padding):
        # Get the dimensions of the input
        n, c, h, w = x.shape
        # Calculate the output dimensions
        oh = (h + 2 * padding - 1) // stride + 1
        ow = (w + 2 * padding - 1) // stride + 1
        # Pad the input
        x = np.pad(
            x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), "constant"
        )
        # Create the output
        dx = np.zeros((n, c, h + 2 * padding, w + 2 * padding))
        # Perform the max pooling
        for i in range(oh):
            for j in range(ow):
                dx[
                    :,
                    :,
                    i * stride : i * stride + 2 * padding + 1,
                    j * stride : j * stride + 2 * padding + 1,
                ] += (
                    x[
                        :,
                        :,
                        i * stride : i * stride + 2 * padding + 1,
                        j * stride : j * stride + 2 * padding + 1,
                    ]
                    == y[:, :, i, j][:, None, None, None]
                ) * dy[
                    :, :, i, j
                ][
                    :, None, None, None
                ]
        # Remove the padding
        dx = dx[:, :, padding:-padding, padding:-padding]
        return dx
