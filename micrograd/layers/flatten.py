import numpy as np

from micrograd.tensors.tensor import Function


class Flatten(Function):
    def __init__(self, input):
        super().__init__([input])
        output_tensor_type = type(input)
        output_shape = (np.prod(input.value.shape),)
        output_dtype = input.value.dtype
        self.output = output_tensor_type(
            value=np.zeros(output_shape, dtype=output_dtype)
        )

    def _forward(self):
        self.output.value = self.flatten(self.input.value)

    def _backward(self):
        if self.input.requires_grad:
            self.input.grad = self.input.grad + self.reshape(
                self.output.grad, self.input.value.shape
            )

    def flatten(self, x):
        return x.reshape(x.shape[0], -1)

    def reshape(self, x, shape):
        return x.reshape(shape)
