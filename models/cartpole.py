# This will be a demo of the custom Tensor library
# We will use cartpole as the example environment

import hashlib
import os
import pickle
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
from micrograd.layers.activations.sigmoid import Sigmoid
from micrograd.utils.debug_utils import debug_print


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, done):
        self.buffer.append((state, action, reward, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class CartPole:
    class Layer:
        def __init__(self):
            self.parameters = []

        def __call__(self, _):
            raise NotImplementedError

        def get_trainable_params(self):
            return self.parameters

    class Linear(Layer):
        def __init__(self, input_dim, output_dim, activation, bias=True):
            self.w = Tensor(np.random.randn(input_dim, output_dim), requires_grad=True)
            self.bias = bias
            if bias:
                self.b = Tensor(np.random.randn(output_dim), requires_grad=True)
            self.activation = activation

            super().__init__()
            self.parameters = [self.w, self.b]

        def __call__(self, x):
            calc = x @ self.w
            if self.bias:
                calc += self.b
            return self.activation(calc)()

    # Reshape layer
    class Reshape(Layer):
        def __init__(self, shape):
            self.shape = shape

            super().__init__()

        def __call__(self, x):
            return x.reshape(self.shape)

    class LSTM(Layer):
        def __init__(
            self,
            num_features,
            hidden_size,
            output_dim,
            activation,
            bias=True,
        ):
            self.activation = activation
            self.hidden_size = hidden_size
            self.bias = bias

            # Initialize weights and biases for the LSTM cell
            self.num_gates = 4
            self.num_layers = 1
            # Have to transpose the weights to match the input shape and perform auto grad
            self.w_ih = Tensor(
                np.random.randn(num_features, self.num_gates * hidden_size),
                requires_grad=True,
            )
            self.w_hh = Tensor(
                np.random.randn(hidden_size, self.num_gates * hidden_size),
                requires_grad=True,
            )
            if bias:
                self.b_ih = (
                    Tensor(
                        np.zeros((self.num_gates * hidden_size,)), requires_grad=True
                    )
                    if bias
                    else None
                )
                self.b_hh = (
                    Tensor(
                        np.zeros((self.num_gates * hidden_size,)), requires_grad=True
                    )
                    if bias
                    else None
                )

            # Initialize the weights and biases for the output layer
            self.w_ho = Tensor(
                np.random.randn(hidden_size, output_dim), requires_grad=True
            )
            if bias:
                self.b_ho = (
                    Tensor(np.zeros((output_dim,)), requires_grad=True)
                    if bias
                    else None
                )

            super().__init__()
            self.parameters = [
                self.w_ih,
                self.w_hh,
                self.w_ho,
            ]
            if bias:
                self.parameters.extend([self.b_ih, self.b_hh, self.b_ho])

        def __call__(self, x):
            # LSTM cell computations
            # Get the batch size dynamically
            batch_size = x.shape[1]
            # Initialize the hidden state dynamically based on the batch size
            if not hasattr(self, "h_prev"):
                self.h_prev = Tensor(
                    np.zeros((self.num_layers, batch_size, self.hidden_size))
                )
            # Initialize the cell state dynamically based on the batch size
            if not hasattr(self, "c_prev"):
                self.c_prev = Tensor(
                    np.zeros((self.num_layers, batch_size, self.hidden_size))
                )

            gi = x @ self.w_ih
            gh = self.h_prev @ self.w_hh
            if self.bias:
                gi += self.b_ih
                gh += self.b_hh

            i_f, i_i, i_c, i_o = Tensor.split(gi, self.num_gates, axis=2)
            h_f, h_i, h_c, h_o = Tensor.split(gh, self.num_gates, axis=2)

            # TODO: Replace np.tanh with function Tanh
            forgetgate = self.activation((i_f + h_f))()
            ingate = self.activation((i_i + h_i))()
            cellgate = (forgetgate * self.c_prev) + (
                ingate * Tensor(np.tanh((i_c + h_c).value))
            )
            self.c_prev = cellgate
            outgate = self.activation((i_o + h_o))()

            h_next = outgate * Tensor(np.tanh(cellgate.value))

            # Output layer computations
            y_pred = h_next @ self.w_ho
            if self.bias:
                y_pred += self.b_ho

            self.h_prev = h_next

            return y_pred

    class Model:
        def __init__(
            self,
            num_features,
            lstm_dim,
            dense_dims,
            output_dim,
            action_space,
            lstm_hidden_activation=Sigmoid,
            hidden_activation=ReLU,
            output_activation=Softmax,
        ):
            # Model architecture
            # Input -> LSTM -> Hidden1 -> Hidden2 -> Output
            batch_size = 1
            seq_len = 1
            self.layers = [
                CartPole.Reshape((seq_len, batch_size, num_features)),
                CartPole.LSTM(
                    num_features,
                    lstm_dim,
                    dense_dims[0],
                    lstm_hidden_activation,
                ),
                CartPole.Linear(dense_dims[0], dense_dims[1], hidden_activation),
                CartPole.Linear(dense_dims[1], output_dim, output_activation),
                CartPole.Reshape((output_dim,)),
            ]
            self.parameters = []
            for layer in self.layers:
                self.parameters.extend(layer.get_trainable_params())
            self.action_space = action_space

            # Create a hash of the model's layers and the input dimensions
            model_parameters = (
                num_features,
                lstm_dim,
                dense_dims[0],
                dense_dims[1],
                output_dim,
                lstm_hidden_activation,
                hidden_activation,
                output_activation,
            )
            self.model_hash = CartPole.Model.create_model_hash(model_parameters)

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

        @staticmethod
        def create_model_hash(model_parameters):
            model_parameters_string = "".join(str(param) for param in model_parameters)
            return hashlib.sha256(model_parameters_string.encode()).hexdigest()

        def save(self, path):
            print(f"Saving model to {path}")
            model_data = {
                "hash": self.model_hash,
                "parameters": [pickle.dumps(param.value) for param in self.parameters],
            }
            np.savez(path, **model_data)

        def load(self, path):
            print(f"Loading model from {path}")
            model_data = np.load(path, allow_pickle=True)
            if model_data["hash"] != self.model_hash:
                raise ValueError("Saved model hash doesn't match current model hash")
            for param, saved_param in zip(self.parameters, model_data["parameters"]):
                param.value = pickle.loads(saved_param)

    class ProximalPolicy:
        # Epsilon-greedy exploration strategy with adaptive epsilon
        def __init__(
            self, model, epsilon=0.1, epsilon_decay=0.99, performance_threshold=0.5
        ):
            self.model = model
            self.epsilon = epsilon
            self.epsilon_decay = epsilon_decay
            self.performance_threshold = performance_threshold

        def __call__(self, model_input, performance):
            if np.random.uniform(0, 1) < self.epsilon:
                # Exploration: choose a random action
                action = self.model.get_random_action()
            else:
                # Exploitation: choose the best action according to the current policy
                action = np.argmax(self.model(model_input).value)
                # Update the performance threshold based on the current performance
                self.performance_threshold = max(
                    self.performance_threshold, performance
                )

            # Adjust epsilon based on performance
            # FIXME: How do I handle the performance threshold?
            if performance > (self.performance_threshold * 0.9):
                self.epsilon *= self.epsilon_decay
            else:
                self.epsilon /= self.epsilon_decay

            # Clip epsilon to be between 0 and 1
            self.epsilon = np.clip(self.epsilon, 0.0, 1.0)

            return action

    def __init__(self, do_load_model=False, do_save_model=False):
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
        self.do_load_model = do_load_model
        self.do_save_model = do_save_model

        self.seed = None
        if self.seed is None:
            self.seed = np.random.randint(0, 1000)
        debug_print(f"Seed: {self.seed}")
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.env = gym.make("CartPole-v1", render_mode="rgb_array")
        self.num_actions = int(np.prod(self.env.action_space.shape))
        self.num_observations = int(np.prod(self.env.observation_space.shape))
        debug_print("Initialized environment")

        self.hyperparameters = {
            "num_history": 4,
            "num_history_features": int(
                self.num_actions + self.num_observations + 1
            ),  # History of observation, action, and reward as well as the current observation
            "base_gamma": 0.99,  # Attenuation factor for the reward
            "base_gamma_decay": 0.999,  # Decay factor for the gamma value
            "target_steps": 200,
            "lr_adjustment_frequency": 999,  # Number of epochs to adjust the learning rate
            "model_lstm_dim": 16,
            "model_dense_dim1": 16,
            "model_dense_dim2": 8,
            "optimizer_lr_max": 1e-5,
            "optimizer_lr_min": 1e-5,
            "optimizer_lr_decay_min": 1.0,
            "optimizer_lr_decay_max": 1.0,
            "lr_adjustment_rate": 1.0,
        }

        # Last n actions, rewards, and gamma values
        self.history = deque(maxlen=self.hyperparameters["num_history"])

        # Init gamma
        # Gamma will be attenuated in order to increase the importance of future rewards as the model learns
        self.gamma = self.hyperparameters["base_gamma"]

        self.reset_env(self.seed, num_episodes=0)
        self.avg_steps = 0.0

        model_num_features = (
            self.hyperparameters["num_history_features"]
            * self.hyperparameters["num_history"]
        )
        model_output_dim = self.num_actions
        self.model = CartPole.Model(
            model_num_features,
            self.hyperparameters["model_lstm_dim"],
            [
                self.hyperparameters["model_dense_dim1"],
                self.hyperparameters["model_dense_dim2"],
            ],
            model_output_dim,
            self.env.action_space,
        )

        # Load the saved model if it exists
        self.load_model()

        # Time decay learning rate, targeting 100 steps to reach the minimum learning rate
        self.optimizer_lr_decay = (
            self.hyperparameters["optimizer_lr_max"]
            - self.hyperparameters["optimizer_lr_min"]
        ) / self.hyperparameters["target_steps"]
        # Set to the middle of the learning rate range
        self.optimizer_lr = (
            self.hyperparameters["optimizer_lr_max"]
            + self.hyperparameters["optimizer_lr_min"]
        ) / 2.0
        # Defining lr_decay not necessary in ctor as it is set in setup_lr_adjustment
        self.optimizer = AdamOptim(
            self.model.get_parameters(),
            lr=self.optimizer_lr,
        )
        self.optimizer.setup_lr_adjustment(
            lr_adjustment_rate=self.hyperparameters["lr_adjustment_rate"],
            optimizer_lr_min=self.hyperparameters["optimizer_lr_min"],
            optimizer_lr_max=self.hyperparameters["optimizer_lr_max"],
            optimizer_lr_decay_min=self.hyperparameters["optimizer_lr_decay_min"],
            optimizer_lr_decay_max=self.hyperparameters["optimizer_lr_decay_max"],
        )

        self.loss = lambda y_pred, y_true: CrossEntropyLoss(y_pred, y_true)()
        self.policy = CartPole.ProximalPolicy(self.model)

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

    def reset_env(self, seed=None, num_episodes=None):
        debug_print("Resetting environment")
        if num_episodes is not None:
            self.num_episodes = num_episodes
        num_steps = self.env._elapsed_steps
        if num_steps is not None and self.mode == "train":
            self.avg_steps = 0.05 * num_steps + (1 - 0.05) * self.avg_steps
            # Adjust the learning rate and decay less frequently
            if self.num_episodes % self.hyperparameters["lr_adjustment_frequency"] == 0:
                self.optimizer.adjust_learning_rate(
                    num_steps, self.avg_steps, self.hyperparameters["target_steps"]
                )
        observation, _ = self.env.reset(seed=seed)
        self.reward = 0
        # Running average of the rewards
        self.running_reward = 0.0
        # Reset the gamma value, help the model keep focus on long term by adjusting the decay based on the delta achieved
        gamma_delta = self.hyperparameters["base_gamma"] - self.gamma
        self.gamma = self.hyperparameters["base_gamma"]
        self.gamma_decay = self.hyperparameters["base_gamma_decay"] - (
            gamma_delta / 2.0
        )
        self.history.clear()
        # Initialize the history with random observations
        for _ in range(self.hyperparameters["num_history"] - 1):
            self.history.append(
                np.random.randn(self.hyperparameters["num_history_features"])
            )
        self.history.append(
            np.concatenate([observation, np.random.randn(self.num_actions + 1)])
        )
        self.state = Tensor(np.concatenate([h for h in self.history]))
        self.model_input = self.state
        self.model_output = None
        self.done = False

    def step(self, action):
        if self.done:
            return
        observation, reward, terminated, truncated, _ = self.env.step(action)
        # Scale reward as we survive more steps
        self.reward = reward / self.gamma
        # Update the running reward
        self.running_reward = 0.05 * self.reward + (1 - 0.05) * self.running_reward
        # Attenuate the gamma value
        self.gamma *= self.gamma_decay
        # Update the history with the current action and reward
        self.history.append(np.concatenate([observation, [action, reward]]))
        self.state = Tensor(np.concatenate([h for h in self.history]))
        self.model_input = self.state
        debug_print(f"Terminated: {terminated}, Truncated: {truncated}")
        self.done = terminated or truncated

    def choose_action(self):
        return self.policy(self.model_input, self.running_reward)

    def get_target(self):
        if self.done:
            return Tensor(self.reward)
        self.model_output = self.model(self.model_input)
        return Tensor(self.reward + (self.gamma * self.running_reward))

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
            self.state, action, self.reward, self.done = sample
            self.update_model(action)

    def save_model(self):
        if self.do_save_model:
            self.model.save(self.model_save_path)

    def load_model(self):
        if self.do_load_model:
            self.model.load(self.model_save_path)

    def train(
        self,
        num_episodes,
        batch_size=16,
        sample_freq=4,
        num_pass_early_stop=10,
    ):
        self.mode = "train"
        buffer = ReplayBuffer(capacity=(batch_size * sample_freq * 4))
        pass_streak = 0
        with Tensor.with_auto_grad(True):
            for _ in tqdm(range(num_episodes)):
                self.reset_env()
                self.num_episodes += 1
                sample_cnt = 0
                while not self.done and (
                    self.env._elapsed_steps < self.hyperparameters["target_steps"]
                ):
                    self.render()
                    action = self.choose_action()
                    buffer.push(self.state, action, self.reward, self.done)
                    self.update_model(action)
                    sample_cnt = (sample_cnt + 1) % sample_freq
                    if (
                        (sample_cnt == 0)
                        and not self.done
                        and (len(buffer) >= batch_size)
                    ):
                        self.update_model_from_buffer(buffer, batch_size)
                if self.env._elapsed_steps < self.hyperparameters["target_steps"]:
                    pass_streak = 0
                else:
                    pass_streak += 1
                    if pass_streak >= num_pass_early_stop:
                        print(
                            f"Early stopping at episode {_} with {pass_streak} consecutive passes"
                        )
                        break

    def test(self, num_episodes):
        self.mode = "test"
        with Tensor.with_auto_grad(False):
            best_num_steps = 0
            pass_fail = []
            frames = []
            for _ in tqdm(range(num_episodes)):
                self.reset_env()
                this_frames = []
                while not self.done and (
                    self.env._elapsed_steps < self.hyperparameters["target_steps"]
                ):
                    frame = self.render()
                    this_frames.append(frame)
                    action = self.choose_action()
                    self.step(action)
                if self.env._elapsed_steps > best_num_steps:
                    best_num_steps = self.env._elapsed_steps
                    if self.save_best_train_frames:
                        frames = this_frames
                if self.env._elapsed_steps < self.hyperparameters["target_steps"]:
                    debug_print("Test episode finished in failure")
                    pass_fail.append(0)
                else:
                    debug_print("Test episode finished in success")
                    pass_fail.append(1)
            passrate = np.mean(pass_fail)
            print(f"Pass rate: {passrate * 100}%")
            print(f"Best number of steps: {best_num_steps}")
            if passrate >= self.save_model_pass_threshold:
                self.save_model()
            if self.save_best_train_frames:
                print(f"Rendering best frames of length: {len(frames)}")
                self.render_saved_frames(frames)


def train_cartpole():
    cart_pole = CartPole(do_load_model=True, do_save_model=True)
    num_episodes_per_round = 100
    num_rounds = 50
    for round_num in range(num_rounds):
        print(f"Training round {round_num}")
        cart_pole.train(num_episodes_per_round)
        cart_pole.test(20)


if __name__ == "__main__":
    print("Starting training and testing")
    train_cartpole()
    print("Training and testing finished")
