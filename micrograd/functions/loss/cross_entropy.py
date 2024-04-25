import numpy as np

from micrograd.tensors.tensor import Function, Tensor


class CrossEntropyLoss(Function):
    def __init__(self, y_pred, y_true):
        super().__init__([y_pred, y_true])

    def cross_entropy(self, y, t):
        return -np.sum(t * np.log(y + 1e-7)) / y.shape[0]

    def cross_entropy_grad(self, y, t, dy):
        return -dy * t / y.shape[0]

    def _forward(self):
        self.output = Tensor(
            self.cross_entropy(self.inputs[0].value, self.inputs[1].value),
            requires_grad=(
                self.inputs[0].requires_grad or self.inputs[1].requires_grad
            ),
        )

    def _backward(self):
        if self.inputs[0].requires_grad:
            self.inputs[0].grad = self.inputs[0].grad + self.cross_entropy_grad(
                self.inputs[0].value, self.inputs[1].value, self.output.grad
            )
        if self.inputs[1].requires_grad:
            self.inputs[1].grad = self.inputs[1].grad + self.cross_entropy_grad(
                self.inputs[1].value, self.inputs[0].value, self.output.grad
            )
