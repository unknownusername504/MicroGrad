import numpy as np

from micrograd.tensors.tensor import Function, Tensor


class MeanSquaredError(Function):
    def __init__(self, y_true, y_pred):
        super().__init__([y_true, y_pred])

    def mean_squared_error(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))

    def mean_squared_error_grad(self, y_true, y_pred, dy):
        return 2 * (y_true - y_pred) * dy / y_true.shape[0]

    def _forward(self):
        self.output = Tensor(
            self.mean_squared_error(self.inputs[0].value, self.inputs[1].value),
            requires_grad=(
                self.inputs[0].requires_grad or self.inputs[1].requires_grad
            ),
        )

    def _backward(self):
        if self.inputs[0].requires_grad:
            self.inputs[0].grad = self.inputs[0].grad + self.mean_squared_error_grad(
                self.inputs[0].value, self.inputs[1].value, self.output.grad
            )
        if self.inputs[1].requires_grad:
            self.inputs[1].grad = self.inputs[1].grad + self.mean_squared_error_grad(
                self.inputs[1].value, self.inputs[0].value, self.output.grad
            )
