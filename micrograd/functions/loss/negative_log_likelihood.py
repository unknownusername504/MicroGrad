import numpy as np

from micrograd.tensors.tensor import Function, Tensor


class NegativeLogLikelihood(Function):
    def __init__(self, action_probs, action, reduction="mean"):
        if reduction not in ["none", "mean", "sum", None]:
            raise ValueError(
                f"Reduction method {reduction} not supported. Use 'mean' or 'sum' or 'none'/None"
            )
        self.reduction = reduction
        super().__init__([action_probs, action], function=type(self).__name__)

    def negative_log_likelihood(self, action_probs, action):
        # Apply log-softmax to action_probs for NLL computation
        log_probs = action_probs - np.log(
            np.sum(np.exp(action_probs), axis=1, keepdims=True)
        )
        return -log_probs[np.arange(action_probs.shape[0]), action]

    def negative_log_likelihood_grad(self, action_probs, action, dy):
        grad = np.exp(action_probs) / np.sum(
            np.exp(action_probs), axis=1, keepdims=True
        )
        for i in range(action_probs.shape[0]):
            grad[i, action[i]] -= 1
        # Ensure dy is a 1D array with shape (N,)
        if np.isscalar(dy):
            dy = np.array([dy], dtype=Tensor.default_dtype)
        elif dy.ndim == 0:
            dy = dy.reshape(1)
        grad *= dy[:, np.newaxis]
        return grad

    def _reduce(self):
        if self.reduction != "none" and self.reduction is not None:
            self.output.reduce(reduction=self.reduction)

    def _forward(self):
        self.output.value = self.negative_log_likelihood(
            self.inputs[0].value, self.inputs[1].value
        )

    def _backward(self):
        if Tensor.get_auto_grad() and self.inputs_requires_grad:
            # Intermediate gradient for the output for reduction
            self.output.grad = np.ones_like(
                self.output.value,
                dtype=Tensor.default_dtype,  # Initialize gradient for output value
            )
            self._reduce()
            if self.inputs[0].requires_grad:
                self.inputs[0].grad += self.negative_log_likelihood_grad(
                    self.inputs[0].value, self.inputs[1].value, self.output.grad
                )
            if self.inputs[1].requires_grad:
                raise NotImplementedError(
                    "Backward pass for the action is not implemented"
                )

            # Final gradients are 1s wrt the input
            self.output.grad = np.ones_like(
                self.output.value,
                dtype=Tensor.default_dtype,
            )
        else:
            self._reduce()
