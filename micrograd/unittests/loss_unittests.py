import unittest
import numpy as np
import torch
from micrograd.tensors.tensor import Tensor
from micrograd.tensors.tensor_i8 import TensorI8
from micrograd.functions.loss.mean_squared_error import MeanSquaredError
from micrograd.functions.loss.negative_log_likelihood import NegativeLogLikelihood

from micrograd.utils.compare import validate_gradients, validate_z_values

# Set default dtype of torch to Tensor.default_dtype
torch.set_default_dtype(
    torch.float32 if (Tensor.default_dtype == np.float32) else torch.float64
)


# Unit tests of operations
class TestActs(unittest.TestCase):
    def loss_unittest_mse(self):
        print("!!! Running loss_unittest_mse !!!")

        # Custom library MSE operation
        with Tensor.with_auto_grad(True):
            y_true = TensorI8([[1, -2], [-3, 4]], requires_grad=True)
            y_pred = TensorI8([[2, -2], [-3, 3]], requires_grad=True)
            mse_func = MeanSquaredError(y_true, y_pred)
            try:
                observed_z = mse_func()
            except Exception as e:
                self.fail(f"Exception occurred: {e}")

        # PyTorch equivalent MSE operation
        y_true_torch = torch.tensor(
            [[1, -2], [-3, 4]], dtype=torch.get_default_dtype(), requires_grad=True
        )
        y_pred_torch = torch.tensor(
            [[2, -2], [-3, 3]], dtype=torch.get_default_dtype(), requires_grad=True
        )
        z_torch = torch.nn.functional.mse_loss(y_true_torch, y_pred_torch)
        z_torch.retain_grad()  # Retain gradients for non-leaf z_torch
        z_torch.sum().backward()  # Summing to ensure backward propagation

        # Validate values
        validate_z_values(observed_z, z_torch, cast_dtype=None)

        # Validate gradients
        validate_gradients(
            [(y_true, y_true_torch), (y_pred, y_pred_torch), (observed_z, z_torch)]
        )

    def loss_unittest_nll(self):
        for reduction in ["none", "mean", "sum"]:
            print(f"!!! Running loss_unittest_nll_{reduction} !!!")

            # Custom library NLL operation
            with Tensor.with_auto_grad(True):
                action_probs = Tensor([[0.3, 0.7], [0.6, 0.4]], requires_grad=True)
                action = Tensor(
                    np.array([1, 0], dtype=np.dtype("long")), requires_grad=False
                )
                nll_func = NegativeLogLikelihood(
                    action_probs, action, reduction=reduction
                )
                try:
                    observed_z = nll_func()
                except Exception as e:
                    self.fail(f"Exception occurred: {e}")

            # PyTorch equivalent NLL operation
            action_probs_torch = torch.tensor(
                [[0.3, 0.7], [0.6, 0.4]],
                dtype=torch.get_default_dtype(),
                requires_grad=True,
            )
            log_softmax_action_probs_torch = torch.nn.functional.log_softmax(
                action_probs_torch, dim=1
            )
            action_torch = torch.tensor(
                [1, 0], dtype=torch.long, requires_grad=False
            )  # Change to long, as this is the expected dtype
            z_torch = torch.nn.functional.nll_loss(
                log_softmax_action_probs_torch, action_torch, reduction=reduction
            )  # Use log probabilities
            z_torch.retain_grad()  # Retain gradients for non-leaf z_torch
            if reduction == "none":
                z_torch.sum().backward()  # Summing to ensure backward propagation
            else:
                z_torch.backward()  # Backward propagation

            # Validate values
            validate_z_values(observed_z, z_torch, cast_dtype=None)

            # Validate gradients
            validate_gradients(
                [
                    (action_probs, action_probs_torch),
                    (observed_z, z_torch),
                ]
            )


if __name__ == "__main__":
    unittest.main()
