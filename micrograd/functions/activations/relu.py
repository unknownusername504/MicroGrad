import numpy as np

from micrograd.tensors.tensor import Function


class ReLU(Function):
    def __init__(self, input):
        super().__init__([input])

    def relu(self, x):
        return np.maximum(x, 0)

    def relu_grad(self, x):
        return np.where(x > 0, 1, 0)

    def _forward(self):
        return self.relu(self.input.value)

    def _backward(self):
        self.input.grad = self.input.grad + self.relu_grad(self.input.value)
