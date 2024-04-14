import numpy as np

from micrograd.tensors.tensor import Function, Tensor


class Softmax(Function):
    def __init__(self, input):
        super().__init__([input])

    def softmax(self, x):
        exp = np.exp(x - np.max(x))
        # FIXME: Why is this necessary?
        axis = 0 if x.ndim == 1 else 1
        return exp / np.sum(exp, axis=axis, keepdims=True)

    def softmax_grad(self, y, dy):
        # FIXME: Why is this necessary?
        axis = 0 if y.ndim == 1 else 1
        return y * (dy - np.sum(y * dy, axis=axis, keepdims=True))

    def _forward(self):
        return Tensor(self.softmax(self.input.value))

    def _backward(self):
        self.input.grad = self.input.grad + self.softmax_grad(
            self.output.value, self.output.grad
        )
