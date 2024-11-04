import numpy as np

from micrograd.tensors.tensor import Function, Tensor


class Sigmoid(Function):
    def __init__(self, input):
        super().__init__(inputs=[input], function=type(self).__name__)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-(x.astype(self.output.value.dtype))))

    def _forward(self):
        self.output.value = self.sigmoid(self.input.value)

    def sigmoid_grad(self):
        return self.output.value * (1 - self.output.value)

    def _backward(self):
        if Tensor.get_auto_grad() and self.inputs_requires_grad:
            self.output.grad = np.ones_like(
                self.output.grad, dtype=Tensor.default_dtype
            )
            self.input.grad += self.output.grad * self.sigmoid_grad()
