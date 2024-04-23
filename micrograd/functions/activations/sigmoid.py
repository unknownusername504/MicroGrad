import numpy as np

from micrograd.tensors.tensor import Function, Tensor


class Sigmoid(Function):
    def __init__(self, input):
        super().__init__([input])

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _forward(self):
        self.output = Tensor(self.sigmoid(self.input.value))

    def sigmoid_grad(self):
        return self.output.value * (1 - self.output.value)

    def _backward(self):
        if self.input.requires_grad:
            self.input.grad = self.input.grad * self.sigmoid_grad()
