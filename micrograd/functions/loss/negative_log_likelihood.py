import numpy as np

from micrograd.tensors.tensor import Function, Tensor


class NegativeLogLikelihood(Function):
    def __init__(self, action_probs, action):
        super().__init__([action_probs, action])

    def negative_log_likelihood(self, action_probs, action):
        return -np.log(action_probs[action])

    def negative_log_likelihood_grad(self, action_probs, action, dy):
        grad = np.zeros_like(action_probs)
        grad[action] = -1 / action_probs[action] * dy
        return grad

    def _forward(self):
        self.output = Tensor(
            self.negative_log_likelihood(self.inputs[0].value, self.inputs[1].value),
            requires_grad=(
                self.inputs[0].requires_grad or self.inputs[1].requires_grad
            ),
        )

    def _backward(self):
        if self.inputs[0].requires_grad:
            self.inputs[0].grad = self.inputs[
                0
            ].grad + self.negative_log_likelihood_grad(
                self.inputs[0].value, self.inputs[1].value, self.output.grad
            )
        if self.inputs[1].requires_grad:
            self.inputs[1].grad = self.inputs[
                1
            ].grad + self.negative_log_likelihood_grad(
                self.inputs[1].value, self.inputs[0].value, self.output.grad
            )
