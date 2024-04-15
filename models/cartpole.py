# This will be a demo of the custom Tensor library
# We will use cartpole as the example environment

import os
import numpy as np
import gym
from collections import deque
import random
import matplotlib.pyplot as plt
import cv2
import imageio
from tqdm import tqdm

from micrograd.tensors.tensor import Tensor

from micrograd.functions.loss.crossentropy import CrossEntropyLoss
from micrograd.functions.optimizers.adam import AdamOptim

from micrograd.layers.activations.relu import ReLU
from micrograd.layers.activations.softmax import Softmax
from micrograd.utils.debug_utils import debug_print


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class CartPole:
    class Model:
        class Layer:
            def __init__(self, input_dim, output_dim, activation):
                self.w = Tensor(np.random.randn(input_dim, output_dim))
                self.b = Tensor(np.random.randn(output_dim))
                self.activation = activation

            def __call__(self, x):
                return self.activation((x @ self.w) + self.b)()

        def __init__(
            self,
            input_dim,
            hidden_dim1,
            hidden_dim2,
            output_dim,
            action_space,
            hidden_activation=ReLU,
            output_activation=Softmax,
        ):
            self.layers = [
                CartPole.Model.Layer(input_dim, hidden_dim1, hidden_activation),
                CartPole.Model.Layer(hidden_dim1, hidden_dim2, hidden_activation),
                CartPole.Model.Layer(hidden_dim2, output_dim, output_activation),
            ]
            self.parameters = [
                param for layer in self.layers for param in [layer.w, layer.b]
            ]
            self.action_space = action_space

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

        def save(self, path):
            np.savez(path, *[param.value for param in self.parameters])

        def load(self, path):
            if not os.path.exists(path):
                print(f"Model file {path} does not exist")
                return
            with np.load(path) as data:
                for param, (_, value) in zip(self.parameters, data.items()):
                    param.value = value

    class ProximalPolicy:
        # Epsilon-greedy exploration strategy
        def __init__(self, model, epsilon=0.1):
            self.model = model
            self.epsilon = epsilon

        def __call__(self, model_input):
            if np.random.uniform(0, 1) < self.epsilon:
                # Exploration: choose a random action
                return self.model.get_random_action()
            else:
                # Exploitation: choose the best action according to the current policy
                return Tensor(np.argmax(self.model(model_input).value)).item()

    def __init__(self):
        self.do_render = False
        self.save_best_train_frames = True

        # Get the file path of the current script
        self.script_path = os.path.realpath(__file__)

        self.model_save_path = os.path.join(
            os.path.dirname(self.script_path), "cartpole_model.npz"
        )
        self.frames_save_path = os.path.join(
            os.path.dirname(self.script_path), "cartpole_best_frames.gif"
        )
        self.save_model_pass_threshold = 0.0
        self.use_saved_model = True

        self.seed = None
        if self.seed is None:
            self.seed = np.random.randint(0, 1000)
        debug_print(f"Seed: {self.seed}")
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.env = gym.make("CartPole-v1", render_mode="rgb_array")
        debug_print("Initialized environment")

        self.reset_env(self.seed)

        model_input_dim = self.env.observation_space.shape[0] * 2
        model_hidden_dim1 = 128
        model_hidden_dim2 = 128
        model_output_dim = self.env.action_space.n
        debug_print(f"Model input dim: {model_input_dim}")
        debug_print(f"Model hidden dim1: {model_hidden_dim1}")
        debug_print(f"Model hidden dim2: {model_hidden_dim2}")
        debug_print(f"Model output dim: {model_output_dim}")
        self.model = CartPole.Model(
            model_input_dim,
            model_hidden_dim1,
            model_hidden_dim2,
            model_output_dim,
            self.env.action_space,
        )
        self.optimizer = AdamOptim(self.model.get_parameters(), lr=1e-4)
        self.loss = lambda y_pred, y_true: CrossEntropyLoss(y_pred, y_true)()
        self.policy = CartPole.ProximalPolicy(self.model)
        self.gamma = 0.99

    def render(self):
        frame = self.env.render()
        if self.done or not self.do_render:
            return frame
        plt.imshow(frame)
        plt.show()
        return frame

    def render_saved_frames(self, frames, fps=30, duration=2):
        # Render the saved frames in a GIF format
        images = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
        # Skip frames to get the desired duration or less
        step = int(len(images) / (fps * duration))
        if step == 0:
            step = 1
        images = images[::step]

        try:
            imageio.mimsave(self.frames_save_path, images, fps=fps)
            print("GIF saved successfully.")
        except Exception as e:
            print(f"Error saving GIF: {e}")

    def reset_env(self, seed=None):
        debug_print("Resetting environment")
        observation, _ = self.env.reset(seed=seed)
        self.reward = 0
        self.state = Tensor(observation)
        self.next_state = self.state
        self.model_input = Tensor(
            np.concatenate((self.state.value, self.next_state.value))
        )
        self.model_output = None
        self.done = False

    def step(self, action):
        if self.done:
            return
        observation, reward, terminated, truncated, _ = self.env.step(action)
        self.reward = reward
        self.state = self.next_state
        self.next_state = Tensor(observation)
        self.model_input = Tensor(
            np.concatenate((self.state.value, self.next_state.value))
        )
        debug_print(f"Terminated: {terminated}, Truncated: {truncated}")
        self.done = terminated or truncated

    def choose_action(self):
        return self.policy(self.model_input)

    def get_target(self):
        self.model_output = self.model(self.model_input)
        return Tensor(
            self.reward
            + (self.gamma * self.model_output.max().item() * (1 - self.done))
        )

    def update_model(self, action):
        self.optimizer.zero_grad()
        self.step(action)
        target = self.get_target()
        # Turn off autograd for the loss calculation
        with Tensor.with_auto_grad(False):
            _ = self.loss(self.model_output, target)
        self.optimizer.step()

    def update_model_from_buffer(self, buffer, batch_size):
        samples = buffer.sample(batch_size)
        for sample in samples:
            self.state, action, self.reward, self.next_state, self.done = sample
            self.update_model(action)

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model.load(path)

    def train(self, num_episodes, batch_size=16, sample_freq=4, steps_to_pass=100):
        buffer = ReplayBuffer(capacity=(batch_size * sample_freq * 4))
        if self.use_saved_model:
            print(f"Loading model from {self.model_save_path}")
            self.load_model(self.model_save_path)
        with Tensor.with_auto_grad(True):
            for _ in tqdm(range(num_episodes)):
                self.reset_env()
                sample_cnt = 0
                while not self.done and (self.env._elapsed_steps < steps_to_pass):
                    self.render()
                    action = self.choose_action()
                    buffer.push(
                        self.state, action, self.reward, self.next_state, self.done
                    )
                    self.update_model(action)
                    sample_cnt = (sample_cnt + 1) % sample_freq
                    if (
                        (sample_cnt == 0)
                        and not self.done
                        and (len(buffer) >= batch_size)
                    ):
                        self.update_model_from_buffer(buffer, batch_size)

    def test(self, num_episodes, steps_to_pass=100, num_pass_early_stop=10):
        with Tensor.with_auto_grad(False):
            best_num_steps = 0
            pass_fail = []
            frames = []
            passing_streak = 0
            for _ in tqdm(range(num_episodes)):
                self.reset_env()
                this_frames = []
                while not self.done and (self.env._elapsed_steps < steps_to_pass):
                    frame = self.render()
                    this_frames.append(frame)
                    action = self.choose_action()
                    self.step(action)
                if self.env._elapsed_steps > best_num_steps:
                    best_num_steps = self.env._elapsed_steps
                    if self.save_best_train_frames:
                        frames = this_frames
                if self.done:
                    debug_print("Test episode finished in failure")
                    pass_fail.append(0)
                    passing_streak = 0
                else:
                    debug_print("Test episode finished in success")
                    pass_fail.append(1)
                    passing_streak += 1
                    if passing_streak >= num_pass_early_stop:
                        print(
                            f"Early stopping at episode {_} with {passing_streak} consecutive passes"
                        )
                        break
            passrate = np.mean(pass_fail)
            print(f"Pass rate: {passrate * 100}%")
            print(f"Best number of steps: {best_num_steps}")
            if passrate >= self.save_model_pass_threshold:
                print(f"Saving model to {self.model_save_path}")
                self.save_model(self.model_save_path)
            if self.save_best_train_frames:
                print(f"Rendering best frames of length: {len(frames)}")
                self.render_saved_frames(frames)


def test_cartpole():
    cart_pole = CartPole()
    cart_pole.train(25000)
    cart_pole.test(20)


if __name__ == "__main__":
    print("Starting training and testing")
    test_cartpole()
