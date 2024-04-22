import numpy as np

from micrograd.tensors.tensor import Function, Tensor


class Softmax(Function):
    def __init__(self, input):
        super().__init__([input])

    def softmax(self, x):
        exp = np.exp(x - np.max(x))
        axis = 0 if x.ndim == 1 else 1
        return exp / np.sum(exp, axis=axis, keepdims=True)

    def _forward(self):
        self.output = Tensor(
            self.softmax(self.input.value), requires_grad=self.input.requires_grad
        )

    def softmax_grad(self, y, dy):
        axis = 0 if y.ndim == 1 else 1
        return y * (dy - np.sum(y * dy, axis=axis, keepdims=True))

    def _backward(self):
        if self.input.requires_grad:
            self.input.grad = self.input.grad + self.softmax_grad(
                self.output.value, self.output.grad
            )
