import numpy as np

from micrograd.tensors.tensor import Function


class Flatten(Function):
    def __init__(self, inputs, output):
        super().__init__(inputs, output)

    def _forward(self):
        self.output.value = self.flatten(self.inputs[0].value)

    def _backward(self):
        self.inputs[0].grad = self.inputs[0].grad + self.reshape(
            self.output.grad, self.inputs[0].value.shape
        )

    def flatten(self, x):
        return x.reshape(x.shape[0], -1)

    def reshape(self, x, shape):
        return x.reshape(shape)
