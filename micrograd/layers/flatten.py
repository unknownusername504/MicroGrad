import numpy as np

from micrograd.tensors.tensor import Function


class Flatten(Function):
    def __init__(self, input):
        super().__init__([input])

    def _forward(self):
        return self.flatten(self.input.value)

    def _backward(self):
        self.input.grad = self.input.grad + self.reshape(
            self.output.grad, self.input.value.shape
        )

    def flatten(self, x):
        return x.reshape(x.shape[0], -1)

    def reshape(self, x, shape):
        return x.reshape(shape)
