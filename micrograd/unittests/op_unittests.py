import unittest
import numpy as np
from micrograd.tensors.tensor import Tensor
from micrograd.tensors.tensor_u8 import TensorU8
from micrograd.scheduler.schedule import WaveRunner


# Unit tests of operations
class TestOps(unittest.TestCase):
    def op_unittest_add(self):
        print("!!! Running op_unittest_add !!!")
        # Test addition
        # Create the input tensors
        x = TensorU8((2, 2), np.array([[1, 2], [3, 4]]))
        y = TensorU8((2, 2), np.array([[5, 6], [7, 8]]))
        expected_z = TensorU8((2, 2), (x.value + y.value))
        add_func = Tensor.Add([x, y])
        with WaveRunner() as wave_runner:
            # Send the function to the wave runner
            try:
                observed_z = wave_runner.send_function(add_func)
            except Exception as e:
                self.fail(f"Exception occurred: {e}")
        # observed_z = add_func.output
        # Print the result
        print("Tensor.Addition result:")
        print("Expected:")
        print(expected_z)
        print("Observed:")
        print(observed_z)
        # Check the result
        self.assertTrue(expected_z == observed_z)

    def op_unittest_sub(self):
        print("!!! Running op_unittest_sub !!!")
        # Test subtraction
        # Create the input tensors
        x = TensorU8((2, 2), np.array([[8, 7], [6, 5]]))
        y = TensorU8((2, 2), np.array([[1, 2], [3, 4]]))
        expected_z = TensorU8((2, 2), (x.value - y.value))
        sub_func = Tensor.Sub([x, y])
        with WaveRunner() as wave_runner:
            # Send the function to the wave runner
            try:
                observed_z = wave_runner.send_function(sub_func)
            except Exception as e:
                self.fail(f"Exception occurred: {e}")
        # observed_z = sub_func.output
        # Print the result
        print("Tensor.Subtraction result:")
        print("Expected:")
        print(expected_z)
        print("Observed:")
        print(observed_z)
        # Check the result
        self.assertTrue(expected_z == observed_z)

    def op_unittest_dot(self):
        print("!!! Running op_unittest_dot !!!")
        # Test dot product
        # Create the input tensors
        x = TensorU8((2, 2), np.array([[1, 2], [3, 4]]))
        y = TensorU8((2, 2), np.array([[5, 6], [7, 8]]))
        expected_z = TensorU8((2, 2), np.dot(x.value, y.value))
        dot_func = Tensor.Dot([x, y])
        with WaveRunner() as wave_runner:
            # Send the function to the wave runner
            try:
                observed_z = wave_runner.send_function(dot_func)
            except Exception as e:
                self.fail(f"Exception occurred: {e}")
        # observed_z = dot_func.output
        # Print the result
        print("Tensor.Dot product result:")
        print("Expected:")
        print(expected_z)
        print("Observed:")
        print(observed_z)
        # Check the result
        self.assertTrue(expected_z == observed_z)

    def op_unittest_mul(self):
        print("!!! Running op_unittest_mul !!!")
        # Test multiplication
        # Create the input tensors
        x = TensorU8((2, 2), np.array([[1, 2], [3, 4]]))
        y = TensorU8((2, 2), np.array([[5, 6], [7, 8]]))
        expected_z = TensorU8((2, 2), np.matmul(x.value, y.value))
        mul_func = Tensor.Matmul([x, y])
        with WaveRunner() as wave_runner:
            # Send the function to the wave runner
            try:
                observed_z = wave_runner.send_function(mul_func)
            except Exception as e:
                self.fail(f"Exception occurred: {e}")
        # observed_z = mul_func.output
        # Print the result
        print("Multiplication result:")
        print("Expected:")
        print(expected_z)
        print("Observed:")
        print(observed_z)
        # Check the result
        self.assertTrue(expected_z == observed_z)
