import numpy as np

from micrograd.tensors.tensor import Function, Tensor


class BinaryCrossEntropyLoss(Function):
    def __init__(self, y_pred, y_true):
        super().__init__([y_pred, y_true], function=type(self).__name__)

    def binary_cross_entropy(self, y, t):
        # Binary Cross Entropy formula: - (t * log(y) + (1 - t) * log(1 - y))
        return -np.mean(t * np.log(y + 1e-7) + (1 - t) * np.log(1 - y + 1e-7))

    def binary_cross_entropy_grad(self, y, t, dy):
        # Gradient of BCE: (y - t) / (y * (1 - y))
        return dy * (y - t) / (y * (1 - y) + 1e-7) / y.shape[0]

    def _forward(self):
        # Calculate the binary cross entropy and store as output
        self.output.value = self.binary_cross_entropy(
            self.inputs[0].value, self.inputs[1].value
        ).astype(self.output.value.dtype)

    def _backward(self):
        if Tensor.get_auto_grad() and self.inputs_requires_grad:
            self.output.grad = np.ones_like(
                self.output.grad, dtype=Tensor.default_dtype
            )
            if self.inputs[0].requires_grad:
                print("input 0 has grads")
                self.inputs[0].grad += self.binary_cross_entropy_grad(
                    self.inputs[0].value, self.inputs[1].value, self.output.grad
                )
            if self.inputs[1].requires_grad:
                print("input 1 has grads")
                self.inputs[1].grad += self.binary_cross_entropy_grad(
                    self.inputs[1].value, self.inputs[0].value, self.output.grad
                )
