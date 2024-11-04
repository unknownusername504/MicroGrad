import numpy as np

from micrograd.tensors.tensor import Function, Tensor


class MeanSquaredError(Function):
    def __init__(self, y_true, y_pred):
        super().__init__([y_true, y_pred], function=type(self).__name__)

    def mean_squared_error(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred).astype(self.output.value.dtype))

    def mean_squared_error_grad(self, y_true, y_pred, dy):
        return (2 * (y_true - y_pred) * dy) / y_true.size

    def _forward(self):
        self.output.value = self.mean_squared_error(
            self.inputs[0].value, self.inputs[1].value
        )

    def _backward(self):
        if Tensor.get_auto_grad() and self.inputs_requires_grad:
            self.output.grad = np.ones_like(
                self.output.grad, dtype=Tensor.default_dtype
            )
            if self.inputs[0].requires_grad:
                self.inputs[0].grad += self.mean_squared_error_grad(
                    self.inputs[0].value, self.inputs[1].value, self.output.grad
                )
            if self.inputs[1].requires_grad:
                self.inputs[1].grad += self.mean_squared_error_grad(
                    self.inputs[1].value, self.inputs[0].value, self.output.grad
                )
