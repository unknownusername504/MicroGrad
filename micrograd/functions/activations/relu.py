import numpy as np

from micrograd.tensors.tensor import Function, Tensor


class ReLU(Function):
    def __init__(self, input):
        super().__init__(inputs=[input], function=type(self).__name__)

    def relu(self, x):
        return np.maximum(x, 0)

    def _forward(self):
        self.output.value = self.relu(self.input.value)

    def relu_grad(self):
        return np.where(self.output.value > 0, 1, 0)

    def _backward(self):
        if Tensor.get_auto_grad() and self.inputs_requires_grad:
            self.output.grad = np.ones_like(
                self.output.grad, dtype=Tensor.default_dtype
            )
            self.input.grad += self.output.grad * self.relu_grad()
