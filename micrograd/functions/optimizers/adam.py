import numpy as np

from micrograd.tensors.tensor import Tensor


class AdamOptim:
    def __init__(self, params, lr=1e-3, lr_decay=None, betas=(0.9, 0.999), eps=1e-8):
        self.params = params
        self.lr = lr
        self.lr_decay = lr_decay
        self.betas = betas
        self.eps = eps
        self.m = [0] * len(params)
        self.v = [0] * len(params)
        self.t = 0

    def zero_grad(self):
        for param in self.params:
            param.zero_grad()

    def reset_lr(self, lr):
        self.lr = lr

    def get_current_lr(self):
        return self.lr

    def reset_lr_decay(self, lr_decay):
        self.lr_decay = lr_decay

    def setup_lr_adjustment(
        self,
        lr_adjustment_rate=1.0,
        optimizer_lr_min=1e-6,
        optimizer_lr_max=1e-3,
        optimizer_lr_decay_min=0.99,
        optimizer_lr_decay_max=0.9,
    ):
        self.lr_adjustment_rate = lr_adjustment_rate
        self.optimizer_lr_min = optimizer_lr_min
        self.optimizer_lr_max = optimizer_lr_max
        self.optimizer_lr_decay_min = optimizer_lr_decay_min
        self.optimizer_lr_decay_max = optimizer_lr_decay_max
        self.reset_lr(self.optimizer_lr_max)
        self.reset_lr_decay(self.optimizer_lr_decay_min)

    def adjust_learning_rate(self, num_steps, avg_steps, target_steps):
        # Evaluate the learning rate and learning rate decay based on the number of steps passed
        current_lr = self.get_current_lr()
        current_lr_decay = self.lr_decay

        # We want to increase the learning rate if the number of steps is less than the average and decrease it otherwise
        lr_delta = self.optimizer_lr_max - current_lr
        lr_decay_delta = self.optimizer_lr_decay_max - current_lr_decay

        # Scale the adjustment by the ratio of remaining steps to target steps
        remaining_steps_ratio = max(0, (target_steps - num_steps) / target_steps)

        # Make the learning rate and decay adjustment more gradual
        scaled_lr_delta = lr_delta * remaining_steps_ratio * self.lr_adjustment_rate
        scaled_lr_decay_delta = (
            lr_decay_delta * remaining_steps_ratio * self.lr_adjustment_rate
        )

        # Allow the learning rate and decay to increase again if the model has fallen into a local minimum
        if num_steps < avg_steps:
            new_lr = current_lr + scaled_lr_delta
            new_lr_decay = current_lr_decay + scaled_lr_decay_delta
        else:
            new_lr = current_lr - scaled_lr_delta
            new_lr_decay = current_lr_decay - scaled_lr_decay_delta

        new_lr = min(max(self.optimizer_lr_min, new_lr), self.optimizer_lr_max)
        new_lr_decay = min(
            max(self.optimizer_lr_decay_min, new_lr_decay), self.optimizer_lr_decay_max
        )
        self.reset_lr(new_lr)
        self.reset_lr_decay(new_lr_decay)

    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            # Check for NaN or Inf gradients
            # if np.isnan(param.grad) or np.isinf(param.grad):
            #    param.grad = 0.0
            # Apply gradient clipping
            # if hasattr(param, "grad_clip"):
            #    param.grad = np.clip(param.grad, -param.grad_clip, param.grad_clip)
            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * param.grad
            self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * param.grad**2
            m_hat = (self.m[i] / (1 - self.betas[0] ** self.t)) + self.eps
            v_hat = (self.v[i] / (1 - self.betas[1] ** self.t)) + self.eps
            # Assert on invalid values
            # assert not np.any(np.isnan(m_hat + v_hat)), "Found NaN in Adam optimizer!"
            # assert not np.any(np.isinf(m_hat + v_hat)), "Found Inf in Adam optimizer!"
            # assert not np.any((m_hat + v_hat) == 0.0), "Found zero in Adam optimizer!"
            result = Tensor((self.lr * m_hat) / np.sqrt(v_hat))
            param -= result
        if self.lr_decay is not None:
            # Apply learning rate decay
            self.lr = max(self.lr * (1 - self.lr_decay), self.optimizer_lr_min)
