import numpy as np

from micrograd.tensors.tensor import Function


class Conv2D(Function):
    def __init__(self, inputs, output, stride, padding):
        super().__init__(inputs, output)
        self.stride = stride
        self.padding = padding

    def _forward(self):
        self.output.value = self.conv2d(
            self.inputs[0].value, self.inputs[1].value, self.stride, self.padding
        )

    def _backward(self):
        self.inputs[0].grad = self.inputs[0].grad + self.conv2d_grad(
            self.inputs[0].value,
            self.inputs[1].value,
            self.output.grad,
            self.stride,
            self.padding,
        )
        self.inputs[1].grad = self.inputs[1].grad + self.conv2d_grad(
            self.inputs[1].value,
            self.inputs[0].value,
            self.output.grad,
            self.stride,
            self.padding,
        )

    def conv2d(self, x, w, stride, padding):
        # Get the dimensions of the input
        n, c, h, w = x.shape
        # Get the dimensions of the filter
        fn, fc, fh, fw = w.shape
        # Calculate the output dimensions
        oh = (h + 2 * padding - fh) // stride + 1
        ow = (w + 2 * padding - fw) // stride + 1
        # Pad the input
        x = np.pad(
            x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), "constant"
        )
        # Create the output
        y = np.zeros((n, fn, oh, ow))
        # Perform the convolution
        for i in range(oh):
            for j in range(ow):
                y[:, :, i, j] = np.sum(
                    x[:, :, i * stride : i * stride + fh, j * stride : j * stride + fw]
                    * w,
                    axis=(2, 3, 4),
                )
        return y

    def conv2d_grad(self, x, w, dy, stride, padding):
        # Get the dimensions of the input
        n, c, h, w = x.shape
        # Get the dimensions of the filter
        fn, fc, fh, fw = w.shape
        # Calculate the output dimensions
        oh = (h + 2 * padding - fh) // stride + 1
        ow = (w + 2 * padding - fw) // stride + 1
        # Pad the input
        x = np.pad(
            x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), "constant"
        )
        # Create the output
        dx = np.zeros((n, c, h + 2 * padding, w + 2 * padding))
        dw = np.zeros((fn, fc, fh, fw))
        # Perform the convolution
        for i in range(oh):
            for j in range(ow):
                dx[
                    :, :, i * stride : i * stride + fh, j * stride : j * stride + fw
                ] += np.sum(w * dy[:, :, i, j][:, None, None, None], axis=1)
                dw += np.sum(
                    x[:, :, i * stride : i * stride + fh, j * stride : j * stride + fw]
                    * dy[:, :, i, j][:, None, None, None],
                    axis=0,
                )
        # Remove the padding
        dx = dx[:, :, padding:-padding, padding:-padding]
        return dx, dw
