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

from micrograd.functions.loss.actor_critic import MeanSquaredError
from micrograd.utils.debug_utils import debug_print

from micrograd.tensors.tensor import Tensor

from micrograd.functions.optimizers.adam import AdamOptim
from micrograd.functions.activations.relu import ReLU
from micrograd.functions.activations.sigmoid import Sigmoid

from micrograd.layers.linear import Linear
from micrograd.layers.lstm import LSTM
from micrograd.layers.reshape import Reshape


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
    class Model:
        def __init__(
            self,
            num_features,
            dense_dims,
            output_dim,
            action_space,
            hidden_activation=ReLU,
            output_activation=Sigmoid,
            seq_len: int = 1,
        ):
            # Model architecture
            self.actor_layers = (
                # Input layer
                [
                    # Flatten the output of the LSTM layer
                    Reshape((1, seq_len * num_features)),
                    # Compress the sequence dimension using a linear layer
                    Linear(
                        seq_len * num_features,
                        dense_dims[0],
                        hidden_activation,
                    ),
                ]
                # Hidden layers
                + [
                    Linear(
                        dense_dims[i],
                        dense_dims[i + 1],
                        hidden_activation,
                    )
                    for i in range(0, len(dense_dims) - 1)
                ]
                # Output layer
                + [
                    Linear(
                        dense_dims[-1],
                        output_dim,
                        output_activation,
                    ),
                    Reshape((output_dim,)),
                ]
            )
            self.parameters = []
            for layer in self.actor_layers:
                self.parameters.extend(layer.get_trainable_params())
            self.action_space = action_space

            # Create a hash of the model's layers and the input dimensions
            model_parameters = (
                (num_features,)
                + tuple(dense_dims)
                + (
                    output_dim,
                    hidden_activation,
                    output_activation,
                    seq_len,
                )
            )
            self.model_hash = CartPole.Model.create_model_hash(model_parameters)

        def __call__(self, x):
            for layer in self.actor_layers:
                x = layer(x)
            return x

        def loss(self, y_true, y_pred):
            return MeanSquaredError(y_true, y_pred)()

        def get_parameters(self):
            return self.parameters

        def set_parameters(self, parameters):
            for param, new_param in zip(self.parameters, parameters):
                param.value = new_param.value

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

    class SoftmaxPolicy:
        def __init__(self):
            pass

        def __call__(self, model_output):
            action_probabilities = self.softmax(model_output)
            action = np.random.choice(len(action_probabilities), p=action_probabilities)
            return action

        @staticmethod
        def softmax(x):
            exp = np.exp(x - np.max(x))
            return exp / np.sum(exp)

    class AdaptiveSoftmaxPolicy:
        def __init__(self, epsilon=0.2, epsilon_decay=0.99, performance_threshold=0.0):
            self.epsilon = epsilon
            self.epsilon_decay = epsilon_decay
            self.performance_threshold = performance_threshold
            self.softmax_simple = CartPole.SoftmaxPolicy()

        def __call__(self, model_output, performance):
            if np.random.uniform(0, 1) < self.epsilon:
                action = self.softmax_simple(model_output)
            else:
                action = np.argmax(model_output)

            # Update the performance threshold based on the current performance
            self.performance_threshold = max(self.performance_threshold, performance)
            if performance >= 1.0:
                # Set epsilon to a low value to avoid exploration
                self.epsilon = 0.01
            else:
                # Adjust epsilon based on performance
                if performance > (self.performance_threshold * 0.9):
                    self.epsilon *= self.epsilon_decay
                else:
                    self.epsilon /= self.epsilon_decay

                # Clip epsilon to be between 0 and 1
                self.epsilon = np.clip(self.epsilon, 0.0, 1.0)

            return action

        @staticmethod
        def softmax(x):
            exp = np.exp(x - np.max(x))
            return exp / np.sum(exp)

    def __init__(self, do_load_model=False, do_save_model=False):
        self.do_render = False
        self.save_best_train_frames = True

        # Get the file path of the current script
        self.script_path = os.path.realpath(__file__)

        # Path will be "'...MicroGrad\\.venv\\Lib\\site-packages\\micrograd\\models\\cartpole\\'"
        # but we want "'...MicroGrad\\micrograd\\models\\cartpole\\'"
        if "site-packages" in self.script_path:
            remove_indexes = [
                self.script_path.index(".venv"),
                self.script_path.index("site-packages"),
            ]
            self.script_path = (
                self.script_path[: remove_indexes[0]]
                + self.script_path[remove_indexes[1] + len("site-packages") + 1 :]
            )
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
        self.num_actions = self.env.action_space.n
        self.num_observations = int(np.prod(self.env.observation_space.shape))
        debug_print("Initialized environment")

        self.hyperparameters = {
            "num_history": 6,
            "num_history_features": int(
                self.num_observations + 2
            ),  # History of observation, action, and reward as well as the current observation
            "target_steps": 100,
            "model_dense_dim1": 64,
            "model_dense_dim2": 16,
            "optimizer_lr": 1e-3,
        }

        # Last n actions, rewards values
        self.history = deque(maxlen=self.hyperparameters["num_history"])

        self.reset_env(self.seed)

        model_output_dim = self.num_actions
        self.model = CartPole.Model(
            self.hyperparameters["num_history_features"],
            [
                self.hyperparameters["model_dense_dim1"],
                self.hyperparameters["model_dense_dim2"],
            ],
            model_output_dim,
            self.env.action_space,
            seq_len=self.hyperparameters["num_history"],
        )

        # Load the saved model if it exists
        self.load_model()

        self.optimizer = AdamOptim(
            self.model.get_parameters(),
            lr=self.hyperparameters["optimizer_lr"],
        )

        self.policy = CartPole.AdaptiveSoftmaxPolicy()

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
        self.reward = Tensor(np.zeros(1))
        # Running average of the steps achieved
        self.running_steps_achieved = 0.0
        self.history.clear()
        # Initialize the history with random observations
        for _ in range(self.hyperparameters["num_history"] - 1):
            self.history.append(
                np.random.randn(self.hyperparameters["num_history_features"])
            )
        self.history.append(np.concatenate([observation, np.random.randn(2)]))
        self.state = Tensor(np.stack([h for h in self.history]))
        self.model_input = self.state
        self.model_output = None
        self.done = False

    def step(self, action):
        if self.done:
            return
        observation, reward, terminated, truncated, _ = self.env.step(action)
        self.reward = Tensor(np.array([reward]))
        # Update the running reward
        per_run_weight = 0.1
        self.running_steps_achieved = (per_run_weight * self.env._elapsed_steps) + (
            (1 - per_run_weight) * self.running_steps_achieved
        )
        # Update the history with the current action and reward
        self.history.append(np.concatenate([observation, [action, reward]]))
        self.state = Tensor(np.stack([h for h in self.history]))
        self.model_input = self.state
        debug_print(f"Terminated: {terminated}, Truncated: {truncated}")
        self.done = terminated or truncated

    def choose_action(self):
        self.performance = (
            self.running_steps_achieved / self.hyperparameters["target_steps"]
        )
        action = self.policy(self.model_output.value, self.performance)
        return action

    def update_model(self, action):
        self.optimizer.zero_grad()
        self.step(action)
        # Turn off autograd for the loss calculation
        with Tensor.with_auto_grad(False):
            _ = self.model.loss(self.model_output, self.reward)
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
        buffer = ReplayBuffer(capacity=(batch_size * sample_freq * 4))
        pass_streak = 0
        with Tensor.with_auto_grad(True):
            for _ in range(num_episodes):
                self.reset_env()
                sample_cnt = 0
                while not self.done and (
                    self.env._elapsed_steps < self.hyperparameters["target_steps"]
                ):
                    self.render()
                    self.model_output = self.model(self.model_input)
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
                    self.model_output = self.model(self.model_input)
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
    cart_pole = CartPole(do_load_model=False, do_save_model=True)
    num_episodes_per_round = 100
    num_rounds = 50
    for round_num in range(num_rounds):
        print(f"Training round {round_num}")
        # Send the model to the wave runner and train mini batches
        cart_pole.train(num_episodes_per_round)
        # Sync and collect the results from the wave runner and run all reduce to update the models across all cores
        cart_pole.test(20)


if __name__ == "__main__":
    print("Starting training and testing")
    train_cartpole()
    print("Training and testing finished")
