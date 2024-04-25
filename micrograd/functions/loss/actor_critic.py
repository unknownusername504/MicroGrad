import numpy as np

from micrograd.tensors.tensor import Function, Tensor
from micrograd.functions.loss.negative_log_likelihood import NegativeLogLikelihood
from micrograd.functions.loss.mean_squared_error import MeanSquaredError


class ActorCritic(Function):
    def __init__(
        self,
        action_probs,
        action,
        reward,
        advantage,
    ):
        super().__init__(
            [
                action_probs,
                action,
                reward,
                advantage,
            ]
        )

    def compute_actor_loss(self, action_probs, action):
        return NegativeLogLikelihood(action_probs, action)()

    def compute_critic_loss(self, reward, advantage):
        return MeanSquaredError(reward, advantage)()

    def _forward(self):
        self.actor_loss = self.compute_actor_loss(self.inputs[0], self.inputs[1])
        self.critic_loss = self.compute_critic_loss(self.inputs[2], self.inputs[3])
        self.output = Tensor(
            self.actor_loss + self.critic_loss,
            requires_grad=True,
        )

    def _backward(self):
        if self.inputs[0].requires_grad:
            self.inputs[0].grad = self.inputs[0].grad + self.actor_loss.grad
        if self.inputs[1].requires_grad:
            self.inputs[1].grad = self.inputs[1].grad + self.actor_loss.grad
        if self.inputs[2].requires_grad:
            self.inputs[2].grad = self.inputs[2].grad + self.critic_loss.grad
        if self.inputs[3].requires_grad:
            self.inputs[3].grad = self.inputs[3].grad + self.critic_loss.grad
