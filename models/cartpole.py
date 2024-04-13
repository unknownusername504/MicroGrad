# This will be a demo of the custom Tensor library
# We will use cartpole as the example environment

import numpy as np
import gym
from collections import deque
import random

from micrograd.tensors.tensor import Tensor

from micrograd.functions.loss.crossentropy import CrossEntropyLoss
from micrograd.functions.optimizers.adam import AdamOptim

from micrograd.functions.activations.relu import ReLU
from micrograd.functions.activations.softmax import Softmax


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(
            *random.sample(self.buffer, batch_size)
        )
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)


class CartPole:
    class Model:
        def __init__(self, input_dim, hidden_dim, output_dim, action_space):
            self.w1 = Tensor(np.random.randn(input_dim, hidden_dim))
            self.b1 = Tensor(np.random.randn(hidden_dim))
            self.w2 = Tensor(np.random.randn(hidden_dim, output_dim))
            self.b2 = Tensor(np.random.randn(output_dim))

            self.parameters = [self.w1, self.b1, self.w2, self.b2]

            self.action_space = action_space

            self.layers = [
                lambda x: ReLU((x @ self.w1) + self.b1)(),
                lambda x: (x @ self.w2) + self.b2,
                lambda x: Softmax(x)(),
            ]

        def __call__(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

        def get_parameters(self):
            return self.parameters

        def zero_grad(self):
            for param in self.parameters:
                param.zero_grad()

        def get_random_action(self):
            return self.action_space.sample()

    class ProximalPolicy:
        # Epsilon-greedy exploration strategy
        def __init__(self, model, epsilon=0.1):
            self.model = model
            self.epsilon = epsilon

        def __call__(self, state):
            if np.random.uniform(0, 1) < self.epsilon:
                # Exploration: choose a random action
                return self.model.get_random_action()
            else:
                # Exploitation: choose the best action according to the current policy
                return np.argmax(self.model(state))

    def __init__(self):
        self.env = gym.make("CartPole-v1")
        self.env.reset()
        self.done = False
        self.model = CartPole.Model(4, 128, 2, self.env.action_space)
        self.optimizer = AdamOptim(self.model.get_parameters())
        self.loss = CrossEntropyLoss()
        self.reward = 0
        self.policy = CartPole.ProximalPolicy(self.model)
        self.gamma = 0.99
        self.test_episodes = 10
        self.state = self.env.reset()
        self.next_state = self.state

    def render(self):
        self.env.render()

    def reset(self):
        self.env.reset()
        self.done = False

    def close(self):
        self.env.close()

    def step(self, action):
        if self.done:
            self.reset()
        _, _, self.done, _ = self.env.step(action)
        return self.done

    def choose_action(self):
        return self.policy(self.state)

    def get_target(self):
        return Tensor(
            self.reward + (self.gamma * self.model(self.next_state) * (1 - self.done))
        )

    def update_model(self):
        action = self.choose_action()
        self.step(action)
        target = self.get_target()
        self.loss(self.model(self.state), target)
        self.optimizer.zero_grad()
        self.optimizer.step()

    def train(self, episodes):
        self.reset()
        for _ in range(episodes):
            self.render()
            self.update_model()
            if self.done:
                print("Train episode finished")
                break

    def test(self):
        self.reset()
        for _ in range(self.test_episodes):
            self.render()
            action = self.choose_action()
            self.step(action)
            if self.done:
                print("Test episode finished")
                break


if __name__ == "__main__":
    env = CartPole()
    env.train(100)
    env.test()
    env.close()
