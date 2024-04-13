import numpy as np

from micrograd.tensors.tensor import Function


class Softmax(Function):
    def __init__(self, input):
        super().__init__([input])

    def softmax(self, x):
        exp = np.exp(x - np.max(x))
        return exp / np.sum(exp, axis=1, keepdims=True)

    def softmax_grad(self, y, dy):
        return y * (dy - np.sum(y * dy, axis=1, keepdims=True))

    def _forward(self):
        return self.softmax(self.input.value)

    def _backward(self):
        self.input.grad = self.input.grad + self.softmax_grad(
            self.output.value, self.output.grad
        )
