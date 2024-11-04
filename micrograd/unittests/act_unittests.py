import unittest
import numpy as np
import torch
from micrograd.tensors.tensor import Tensor
from micrograd.tensors.tensor_i8 import TensorI8
from micrograd.functions.activations.relu import ReLU
from micrograd.functions.activations.sigmoid import Sigmoid
from micrograd.functions.activations.softmax import Softmax

from micrograd.utils.compare import validate_gradients, validate_z_values

# Set default dtype of torch to Tensor.default_dtype
torch.set_default_dtype(
    torch.float32 if (Tensor.default_dtype == np.float32) else torch.float64
)


# Unit tests of operations
class TestActs(unittest.TestCase):
    def act_unittest_relu(self):
        print("!!! Running act_unittest_relu !!!")

        # Custom library ReLU operation
        with Tensor.with_auto_grad(True):
            x = TensorI8([[1, -2], [-3, 4]], requires_grad=True)
            relu_func = ReLU(x)
            try:
                observed_z = relu_func()
            except Exception as e:
                self.fail(f"Exception occurred: {e}")

        # PyTorch equivalent ReLU operation
        x_torch = torch.tensor(
            [[1, -2], [-3, 4]], dtype=torch.get_default_dtype(), requires_grad=True
        )
        z_torch = torch.nn.functional.relu(x_torch)
        z_torch.retain_grad()  # Retain gradients for non-leaf z_torch
        z_torch.sum().backward()  # Summing to ensure backward propagation

        # Validate values
        validate_z_values(observed_z, z_torch, cast_dtype=np.int8)

        # Validate gradients
        validate_gradients([(x, x_torch), (observed_z, z_torch)])

    def act_unittest_sigmoid(self):
        print("!!! Running act_unittest_sigmoid !!!")

        # Custom library ReLU operation
        with Tensor.with_auto_grad(True):
            x = TensorI8([[1, -2], [-3, 4]], requires_grad=True)
            relu_func = Sigmoid(x)
            try:
                observed_z = relu_func()
            except Exception as e:
                self.fail(f"Exception occurred: {e}")

        # PyTorch equivalent ReLU operation
        x_torch = torch.tensor(
            [[1, -2], [-3, 4]], dtype=torch.get_default_dtype(), requires_grad=True
        )
        z_torch = torch.nn.functional.sigmoid(x_torch)
        z_torch.retain_grad()  # Retain gradients for non-leaf z_torch
        z_torch.sum().backward()  # Summing to ensure backward propagation

        # Validate values
        validate_z_values(observed_z, z_torch, cast_dtype=None)

        # Validate gradients
        validate_gradients([(x, x_torch), (observed_z, z_torch)])

    def act_unittest_softmax(self):
        print("!!! Running act_unittest_softmax !!!")

        # Custom library ReLU operation
        with Tensor.with_auto_grad(True):
            x = TensorI8([[1, -2], [-3, 4]], requires_grad=True)
            relu_func = Softmax(x)
            try:
                observed_z = relu_func()
            except Exception as e:
                self.fail(f"Exception occurred: {e}")

        # PyTorch equivalent ReLU operation
        x_torch = torch.tensor(
            [[1, -2], [-3, 4]], dtype=torch.get_default_dtype(), requires_grad=True
        )
        z_torch = torch.nn.functional.softmax(x_torch, dim=1)
        z_torch.retain_grad()  # Retain gradients for non-leaf z_torch
        z_torch.sum().backward()  # Summing to ensure backward propagation

        # Validate values
        validate_z_values(observed_z, z_torch, cast_dtype=None)

        # Validate gradients
        validate_gradients([(x, x_torch), (observed_z, z_torch)])


if __name__ == "__main__":
    unittest.main()
