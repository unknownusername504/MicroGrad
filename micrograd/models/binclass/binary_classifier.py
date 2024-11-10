import numpy as np
import hashlib
import pickle

from micrograd.layers.linear import Linear
from micrograd.layers.reshape import Reshape

from micrograd.functions.optimizers.adam import AdamOptim
from micrograd.functions.activations.relu import ReLU
from micrograd.functions.activations.sigmoid import Sigmoid

from micrograd.functions.loss.binary_cross_entropy import BinaryCrossEntropyLoss


from tqdm import tqdm

from micrograd.utils.debug_utils import debug_print

from micrograd.tensors.tensor import Tensor

import os

cur_dir = os.path.dirname(__file__)
data_path = os.path.join(cur_dir, "data.npy")
labels_path = os.path.join(cur_dir, "labels.npy")


class BinaryClassifier:
    def __init__(
        self,
        num_features,
        dense_dims,
        output_dim,
        hidden_activation=ReLU,
        output_activation=Sigmoid,
    ):
        # Model architecture
        self.actor_layers = (
            # Input layer
            [
                # Flatten the output of the LSTM layer
                Reshape((1, num_features)),
                # Compress the sequence dimension using a linear layer
                Linear(
                    num_features,
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

        # Create a hash of the model's layers and the input dimensions
        model_parameters = (
            (num_features,)
            + tuple(dense_dims)
            + (
                output_dim,
                hidden_activation,
                output_activation,
            )
        )
        self.model_hash = BinaryClassifier.create_model_hash(model_parameters)

        self.optimizer = AdamOptim(self.parameters, lr=0.01)

    def __call__(self, x):
        for layer in self.actor_layers:
            x = layer(x)
        return x

    def loss(self, y_pred, y_true):
        return BinaryCrossEntropyLoss(y_pred, y_true)()

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

    def step(self, model_input, model_target):
        self.optimizer.zero_grad()
        model_output = self(model_input)
        loss = self.loss(model_output, model_target)
        self.optimizer.step()
        return loss

    def train(
        self,
        data,
        labels,
        num_episodes,
        batch_size=1,  # FIXME: Model does not support batch training
    ):
        num_batches = len(data) // batch_size
        losses = []
        with Tensor.with_auto_grad(True):
            for _ in tqdm(range(num_episodes)):
                for batch_num in range(num_batches):
                    batch_end = min((batch_num + 1) * batch_size, len(data))
                    batch_start = batch_end - batch_size
                    batch_data = data[batch_start:batch_end]
                    batch_labels = labels[batch_start:batch_end]
                    model_input = Tensor(batch_data)
                    model_target = Tensor(batch_labels)
                    loss = self.step(model_input, model_target)
                    debug_print(f"Loss: {loss}")
                    losses.append(loss.item())
        return sum(losses) / len(losses)


def train_binary_classifier():
    data = np.load(data_path)
    labels = np.load(labels_path)
    num_features = data.shape[1]
    output_dim = labels.shape[1]
    binary_classifier = BinaryClassifier(
        num_features=num_features, dense_dims=[16, 16], output_dim=output_dim
    )
    num_episodes_per_round = 20
    num_rounds = 10
    for round_num in range(num_rounds):
        print(f"Training round {round_num}")
        # Send the model to the wave runner and train mini batches
        avg_loss = binary_classifier.train(data, labels, num_episodes_per_round)
        print(f"Average loss: {avg_loss}")


if __name__ == "__main__":
    print("Starting training and testing")
    train_binary_classifier()
    print("Training and testing finished")
