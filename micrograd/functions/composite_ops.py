import numpy as np

from micrograd.functions.wave_process import Function
from micrograd.functions.primitive_ops import Add, Sub, Dot, Matmul


class Sigmoid(Function):
    def __init__(self, inputs, output):
        super().__init__(inputs, output)

    def forward(self):
        self.output.value = self.sigmoid(self.inputs[0].value)

    def backward(self):
        self.inputs[0].grad = self.add(
            self.inputs[0].grad,
            self.mul(self.output.value, self.sub(1, self.output.value)),
        )

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


class ReLU(Function):
    def __init__(self, inputs, output):
        super().__init__(inputs, output)

    def forward(self):
        self.output.value = self.relu(self.inputs[0].value)

    def backward(self):
        self.inputs[0].grad = self.add(
            self.inputs[0].grad, self.relu_grad(self.inputs[0].value)
        )

    def relu(self, x):
        return np.maximum(x, 0)

    def relu_grad(self, x):
        return np.where(x > 0, 1, 0)


class Softmax(Function):
    def __init__(self, inputs, output):
        super().__init__(inputs, output)

    def forward(self):
        self.output.value = self.softmax(self.inputs[0].value)

    def backward(self):
        self.inputs[0].grad = self.add(
            self.inputs[0].grad, self.softmax_grad(self.output.value, self.output.grad)
        )

    def softmax(self, x):
        exp = np.exp(x - np.max(x))
        return exp / np.sum(exp, axis=1, keepdims=True)

    def softmax_grad(self, y, dy):
        return y * (dy - np.sum(y * dy, axis=1, keepdims=True))


class CrossEntropy(Function):
    def __init__(self, inputs, output):
        super().__init__(inputs, output)

    def forward(self):
        self.output.value = self.cross_entropy(
            self.inputs[0].value, self.inputs[1].value
        )

    def backward(self):
        self.inputs[0].grad = self.add(
            self.inputs[0].grad,
            self.cross_entropy_grad(
                self.inputs[0].value, self.inputs[1].value, self.output.grad
            ),
        )
        self.inputs[1].grad = self.add(
            self.inputs[1].grad,
            self.cross_entropy_grad(
                self.inputs[1].value, self.inputs[0].value, self.output.grad
            ),
        )

    def cross_entropy(self, y, t):
        return -np.sum(t * np.log(y + 1e-7)) / y.shape[0]

    def cross_entropy_grad(self, y, t, dy):
        return -dy * t / y.shape[0]


class Conv2D(Function):
    def __init__(self, inputs, output, stride, padding):
        super().__init__(inputs, output)
        self.stride = stride
        self.padding = padding

    def forward(self):
        self.output.value = self.conv2d(
            self.inputs[0].value, self.inputs[1].value, self.stride, self.padding
        )

    def backward(self):
        self.inputs[0].grad = self.add(
            self.inputs[0].grad,
            self.conv2d_grad(
                self.inputs[0].value,
                self.inputs[1].value,
                self.output.grad,
                self.stride,
                self.padding,
            ),
        )
        self.inputs[1].grad = self.add(
            self.inputs[1].grad,
            self.conv2d_grad(
                self.inputs[1].value,
                self.inputs[0].value,
                self.output.grad,
                self.stride,
                self.padding,
            ),
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


class MaxPool2D(Function):
    def __init__(self, inputs, output, stride, padding):
        super().__init__(inputs, output)
        self.stride = stride
        self.padding = padding

    def forward(self):
        self.output.value = self.max_pool2d(
            self.inputs[0].value, self.stride, self.padding
        )

    def backward(self):
        self.inputs[0].grad = self.add(
            self.inputs[0].grad,
            self.max_pool2d_grad(
                self.inputs[0].value,
                self.output.value,
                self.output.grad,
                self.stride,
                self.padding,
            ),
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


class AvgPool2D(Function):
    def __init__(self, inputs, output, stride, padding):
        super().__init__(inputs, output)
        self.stride = stride
        self.padding = padding

    def forward(self):
        self.output.value = self.avg_pool2d(
            self.inputs[0].value, self.stride, self.padding
        )

    def backward(self):
        self.inputs[0].grad = self.add(
            self.inputs[0].grad,
            self.avg_pool2d_grad(
                self.inputs[0].value, self.output.grad, self.stride, self.padding
            ),
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


class Flatten(Function):
    def __init__(self, inputs, output):
        super().__init__(inputs, output)

    def forward(self):
        self.output.value = self.flatten(self.inputs[0].value)

    def backward(self):
        self.inputs[0].grad = self.add(
            self.inputs[0].grad,
            self.reshape(self.output.grad, self.inputs[0].value.shape),
        )

    def flatten(self, x):
        return x.reshape(x.shape[0], -1)

    def reshape(self, x, shape):
        return x.reshape(shape)
