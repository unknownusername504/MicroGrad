import unittest
import numpy as np
import torch
from micrograd.tensors.tensor import Tensor
from micrograd.tensors.tensor_u8 import TensorU8

# Set default dtype of torch to Tensor.default_dtype
torch.set_default_dtype(
    torch.float32 if (Tensor.default_dtype == np.float32) else torch.float64
)


def assert_array_almost_equal(a, b, atol=1e-6):
    # Check if the dtype is the same
    assert a.dtype == b.dtype, f"Data types do not match: a:{a.dtype} != b:{b.dtype}"
    # Check if the shape is the same
    assert a.shape == b.shape, f"Shapes do not match: a:{a.shape} != b:{b.shape}"
    # Check if the values are the same
    assert np.allclose(a, b, atol=atol), f"Values do not match: a:{a} != b:{b}"


def validate_z_values(observed_z, z_torch, cast_dtype=np.uint8):
    observed_z_np = observed_z.get_value()
    z_torch_np = z_torch.detach().numpy().astype(cast_dtype)

    # Validate values
    assert_array_almost_equal(observed_z_np, z_torch_np)


def validate_gradients(x, y, x_torch, y_torch):
    # Validate gradients
    assert_array_almost_equal(x.grad, x_torch.grad.numpy())
    if (y is not None) and (y_torch is not None):
        assert_array_almost_equal(y.grad, y_torch.grad.numpy())


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
        z_torch.sum().backward()  # Summing to ensure backward propagation

        # Validate values
        validate_z_values(observed_z, z_torch)

        # Validate gradients
        validate_gradients(x, y, x_torch, y_torch)

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
        z_torch.sum().backward()  # Summing to ensure backward propagation

        # Validate values
        validate_z_values(observed_z, z_torch)

        # Validate gradients
        validate_gradients(x, y, x_torch, y_torch)

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
        z_torch.backward()  # Summing to ensure backward propagation

        # Validate values
        validate_z_values(observed_z, z_torch)

        # Validate gradients
        validate_gradients(x, y, x_torch, y_torch)

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
        z_torch.sum().backward()

        # Validate values
        validate_z_values(observed_z, z_torch)

        # Validate gradients
        validate_gradients(x, y, x_torch, y_torch)

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
        z_torch.sum().backward()

        # Validate values
        validate_z_values(observed_z, z_torch, cast_dtype=Tensor.default_dtype)

        # Validate gradients
        validate_gradients(x, y, x_torch, y_torch)

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
        # z_torch.sum().backward()

        # Validate values
        validate_z_values(observed_z, z_torch)

        # Validate gradients
        # validate_gradients(x, y, x_torch, y_torch)

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
        z_torch.sum().backward()

        # Validate values
        validate_z_values(observed_z, z_torch)

        # Validate gradients
        validate_gradients(x, None, x_torch, None)


if __name__ == "__main__":
    unittest.main()
