import unittest
import numpy as np
import torch
from micrograd.tensors.tensor import Tensor
from micrograd.tensors.tensor_u8 import TensorU8

from micrograd.utils.compare import validate_gradients, validate_z_values

# Set default dtype of torch to Tensor.default_dtype
torch.set_default_dtype(
    torch.float32 if (Tensor.default_dtype == np.float32) else torch.float64
)


# Unit tests of operations
class TestOps(unittest.TestCase):
    def op_unittest_add(self):
        print("!!! Running op_unittest_add !!!")

        # Micrograd Tensor operation
        with Tensor.with_auto_grad(True):
            x = TensorU8([[1, 2], [3, 4]], requires_grad=True)
            y = TensorU8([[5, 6], [7, 8]], requires_grad=True)
            add_func = Tensor.Add([x, y])
            try:
                observed_z = add_func()
            except Exception as e:
                self.fail(f"Exception occurred: {e}")

        # PyTorch equivalent operation
        x_torch = torch.tensor(
            [[1, 2], [3, 4]], dtype=torch.get_default_dtype(), requires_grad=True
        )
        y_torch = torch.tensor(
            [[5, 6], [7, 8]], dtype=torch.get_default_dtype(), requires_grad=True
        )
        z_torch = x_torch + y_torch
        z_torch.retain_grad()  # Retain gradients for non-leaf z_torch
        z_torch.sum().backward()  # Summing to ensure backward propagation

        # Validate values
        validate_z_values(observed_z, z_torch, cast_dtype=np.uint8)

        # Validate gradients
        validate_gradients([(x, x_torch), (y, y_torch), (observed_z, z_torch)])

    def op_unittest_sub(self):
        print("!!! Running op_unittest_sub !!!")

        # Micrograd Tensor operation
        with Tensor.with_auto_grad(True):
            x = TensorU8([[8, 7], [6, 5]], requires_grad=True)
            y = TensorU8([[1, 2], [3, 4]], requires_grad=True)
            sub_func = Tensor.Sub([x, y])
            try:
                observed_z = sub_func()
            except Exception as e:
                self.fail(f"Exception occurred: {e}")

        # PyTorch equivalent operation
        x_torch = torch.tensor(
            [[8, 7], [6, 5]], dtype=torch.get_default_dtype(), requires_grad=True
        )
        y_torch = torch.tensor(
            [[1, 2], [3, 4]], dtype=torch.get_default_dtype(), requires_grad=True
        )
        z_torch = x_torch - y_torch
        z_torch.retain_grad()  # Retain gradients for non-leaf z_torch
        z_torch.sum().backward()  # Summing to ensure backward propagation

        # Validate values
        validate_z_values(observed_z, z_torch, cast_dtype=np.uint8)

        # Validate gradients
        validate_gradients([(x, x_torch), (y, y_torch), (observed_z, z_torch)])

    def op_unittest_dot(self):
        print("!!! Running op_unittest_dot !!!")

        # Micrograd Tensor operation
        with Tensor.with_auto_grad(True):
            x = TensorU8([1, 2, 3, 4], requires_grad=True)
            y = TensorU8([5, 6, 7, 8], requires_grad=True)
            dot_func = Tensor.Dot([x, y])
            try:
                observed_z = dot_func()
            except Exception as e:
                self.fail(f"Exception occurred: {e}")

        # PyTorch equivalent operation
        x_torch = torch.tensor(
            [1, 2, 3, 4], dtype=torch.get_default_dtype(), requires_grad=True
        )
        y_torch = torch.tensor(
            [5, 6, 7, 8], dtype=torch.get_default_dtype(), requires_grad=True
        )
        z_torch = torch.dot(x_torch, y_torch)
        z_torch.retain_grad()  # Retain gradients for non-leaf z_torch
        z_torch.backward()  # Backward propagation

        # Validate values
        validate_z_values(observed_z, z_torch, cast_dtype=np.uint8)

        # Validate gradients
        validate_gradients([(x, x_torch), (y, y_torch), (observed_z, z_torch)])

    def op_unittest_mul(self):
        print("!!! Running op_unittest_mul !!!")

        # Micrograd Tensor operation
        with Tensor.with_auto_grad(True):
            x = TensorU8([[1, 2], [3, 4]], requires_grad=True)
            y = TensorU8([[5, 6], [7, 8]], requires_grad=True)
            mul_func = Tensor.Mul([x, y])
            try:
                observed_z = mul_func()
            except Exception as e:
                self.fail(f"Exception occurred: {e}")

        # PyTorch equivalent operation
        x_torch = torch.tensor(
            [[1, 2], [3, 4]], dtype=torch.get_default_dtype(), requires_grad=True
        )
        y_torch = torch.tensor(
            [[5, 6], [7, 8]], dtype=torch.get_default_dtype(), requires_grad=True
        )
        z_torch = x_torch * y_torch
        z_torch.retain_grad()  # Retain gradients for non-leaf z_torch
        z_torch.sum().backward()  # Summing to ensure backward propagation

        # Validate values
        validate_z_values(observed_z, z_torch, cast_dtype=np.uint8)

        # Validate gradients
        validate_gradients([(x, x_torch), (y, y_torch), (observed_z, z_torch)])

    def op_unittest_div(self):
        print("!!! Running op_unittest_div !!!")

        # Micrograd Tensor operation
        with Tensor.with_auto_grad(True):
            x = TensorU8([[8, 7], [6, 5]], requires_grad=True)
            y = TensorU8([[1, 2], [3, 4]], requires_grad=True)
            div_func = Tensor.Div([x, y])
            try:
                observed_z = div_func()
            except Exception as e:
                self.fail(f"Exception occurred: {e}")

        # PyTorch equivalent operation
        x_torch = torch.tensor(
            [[8, 7], [6, 5]], dtype=torch.get_default_dtype(), requires_grad=True
        )
        y_torch = torch.tensor(
            [[1, 2], [3, 4]], dtype=torch.get_default_dtype(), requires_grad=True
        )
        z_torch = x_torch / y_torch
        z_torch.retain_grad()  # Retain gradients for non-leaf z_torch
        z_torch.sum().backward()  # Summing to ensure backward propagation

        # Validate values
        validate_z_values(observed_z, z_torch, cast_dtype=Tensor.default_dtype)

        # Validate gradients
        validate_gradients([(x, x_torch), (y, y_torch), (observed_z, z_torch)])

    def op_unittest_int_div(self):
        print("!!! Running op_unittest_int_div !!!")

        # Micrograd Tensor operation
        with Tensor.with_auto_grad(False):
            x = TensorU8([[8, 7], [6, 5]], requires_grad=False)
            y = TensorU8([[1, 2], [3, 4]], requires_grad=False)
            int_div_func = Tensor.IntDiv([x, y])
            try:
                observed_z = int_div_func()
            except Exception as e:
                self.fail(f"Exception occurred: {e}")

        # PyTorch equivalent operation
        x_torch = torch.tensor(
            [[8, 7], [6, 5]], dtype=torch.get_default_dtype(), requires_grad=False
        )
        y_torch = torch.tensor(
            [[1, 2], [3, 4]], dtype=torch.get_default_dtype(), requires_grad=False
        )
        z_torch = x_torch // y_torch
        # z_torch.retain_grad()  # Retain gradients for non-leaf z_torch
        # z_torch.sum().backward()  # Summing to ensure backward propagation

        # Validate values
        validate_z_values(observed_z, z_torch, cast_dtype=np.uint8)

        # Validate gradients
        # validate_gradients([(x, x_torch), (y, y_torch), (observed_z, z_torch)])

    def op_unittest_neg(self):
        print("!!! Running op_unittest_neg !!!")

        # Micrograd Tensor operation
        with Tensor.with_auto_grad(True):
            x = TensorU8([[1, 2], [3, 4]], requires_grad=True)
            neg_func = Tensor.Neg([x])
            try:
                observed_z = neg_func()
            except Exception as e:
                self.fail(f"Exception occurred: {e}")

        # PyTorch equivalent operation
        x_torch = torch.tensor(
            [[1, 2], [3, 4]], dtype=torch.get_default_dtype(), requires_grad=True
        )
        z_torch = -x_torch
        z_torch.retain_grad()  # Retain gradients for non-leaf z_torch
        z_torch.sum().backward()  # Summing to ensure backward propagation

        # Validate values
        validate_z_values(observed_z, z_torch, cast_dtype=np.uint8)

        # Validate gradients
        validate_gradients([(x, x_torch), (observed_z, z_torch)])


if __name__ == "__main__":
    unittest.main()
