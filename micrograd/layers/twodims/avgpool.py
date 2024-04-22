import numpy as np

from micrograd.tensors.tensor import Function


class AvgPool2D(Function):
    def __init__(self, inputs, output, stride, padding):
        super().__init__(inputs, output)
        self.stride = stride
        self.padding = padding
        # TODO: Preallocate the output tensor

    def _forward(self):
        self.output.value = self.avg_pool2d(
            self.inputs[0].value, self.stride, self.padding
        )

    def _backward(self):
        if self.inputs[0].requires_grad:
            self.inputs[0].grad = self.inputs[0].grad + self.avg_pool2d_grad(
                self.inputs[0].value, self.output.grad, self.stride, self.padding
            )

    def avg_pool2d(self, x, stride, padding):
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
        # Perform the average pooling
        for i in range(oh):
            for j in range(ow):
                y[:, :, i, j] = np.mean(
                    x[
                        :,
                        :,
                        i * stride : i * stride + 2 * padding + 1,
                        j * stride : j * stride + 2 * padding + 1,
                    ],
                    axis=(2, 3),
                )
        return y

    def avg_pool2d_grad(self, x, dy, stride, padding):
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
        # Perform the average pooling
        for i in range(oh):
            for j in range(ow):
                dx[
                    :,
                    :,
                    i * stride : i * stride + 2 * padding + 1,
                    j * stride : j * stride + 2 * padding + 1,
                ] += (
                    dy[:, :, i, j][:, None, None, None] / (2 * padding + 1) ** 2
                )
        # Remove the padding
        dx = dx[:, :, padding:-padding, padding:-padding]
        return dx
