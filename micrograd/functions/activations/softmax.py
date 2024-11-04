import numpy as np

from micrograd.tensors.tensor import Function, Tensor


class Softmax(Function):
    def __init__(self, input, axis=None):
        super().__init__(inputs=[input], function=type(self).__name__)
        self.axis = axis if axis is not None else (0 if (self.input.ndim == 1) else 1)

    def softmax(self, x):
        exp = np.exp(x.astype(self.output.value.dtype) - np.max(x))
        return exp / np.sum(exp, axis=self.axis, keepdims=True)

    def _forward(self):
        self.output.value = self.softmax(self.input.value)

    def softmax_grad(self, y, dy):
        return y * (dy - np.sum(y * dy, axis=self.axis, keepdims=True))

    def _backward(self):
        if Tensor.get_auto_grad() and self.inputs_requires_grad:
            self.output.grad = np.ones_like(
                self.output.grad, dtype=Tensor.default_dtype
            )
            self.input.grad += self.output.grad * self.softmax_grad(
                self.output.value, self.output.grad
            )
