import numpy as np

from micrograd.tensors.tensor import Function


class Sigmoid(Function):
    def __init__(self, input):
        super().__init__([input])

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _forward(self):
        return self.sigmoid(self.input.value)

    def _backward(self):
        self.input.grad = self.input.grad - (self.output.value @ self.output.value)
