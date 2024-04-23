import numpy as np

from micrograd.tensors.tensor import Function, Tensor


class ReLU(Function):
    def __init__(self, input):
        super().__init__([input])

    def relu(self, x):
        return np.maximum(x, 0)

    def _forward(self):
        self.output = Tensor(
            self.relu(self.input.value), requires_grad=self.input.requires_grad
        )

    def relu_grad(self):
        return np.where(self.output.value > 0, 1, 0)

    def _backward(self):
        if self.input.requires_grad:
            self.input.grad = self.input.grad * self.relu_grad()
