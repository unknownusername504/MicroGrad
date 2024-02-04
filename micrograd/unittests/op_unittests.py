import unittest
import numpy as np
from micrograd.functions.primitive_ops import Add, Sub, Dot, Matmul
from micrograd.tensors.tensor_u8 import TensorU8
from micrograd.functions.wave_process import WaveRunner


# Unit tests of operations
class TestOps(unittest.TestCase):
    def op_unittest_add(self):
        # Test addition
        # Create the input tensors
        x = TensorU8((2, 2), np.array([[1, 2], [3, 4]]))
        y = TensorU8((2, 2), np.array([[5, 6], [7, 8]]))
        expected_z = TensorU8((2, 2), np.array([[6, 8], [10, 12]]))
        add_func = Add([x, y])
        with WaveRunner() as wave_runner:
            # Send the function to the wave runner
            try:
                observed_z = wave_runner.send_function(add_func)
            except Exception as e:
                self.fail(f"Exception occurred: {e}")
        # observed_z = add_func.output
        # Print the result
        print("Addition result:")
        print("Expected:")
        print(expected_z)
        print("Observed:")
        print(observed_z)
        # Check the result
        self.assertTrue(expected_z == observed_z)

    def op_unittest_sub(self):
        # Test subtraction
        # Create the input tensors
        x = TensorU8((2, 2), np.array([[8, 7], [6, 5]]))
        y = TensorU8((2, 2), np.array([[1, 2], [3, 4]]))
        expected_z = TensorU8((2, 2), np.array([[7, 5], [3, 1]]))
        sub_func = Sub([x, y])
        with WaveRunner() as wave_runner:
            # Send the function to the wave runner
            try:
                observed_z = wave_runner.send_function(sub_func)
            except Exception as e:
                self.fail(f"Exception occurred: {e}")
        # observed_z = sub_func.output
        # Print the result
        print("Subtraction result:")
        print("Expected:")
        print(expected_z)
        print("Observed:")
        print(observed_z)
        # Check the result
        self.assertTrue(expected_z == observed_z)

    def op_unittest_dot(self):
        # Test dot product
        # Create the input tensors
        x = TensorU8((2, 2), np.array([[1, 2], [3, 4]]))
        y = TensorU8((2, 2), np.array([[5, 6], [7, 8]]))
        expected_z = TensorU8((2, 2), np.array([[19, 22], [43, 50]]))
        dot_func = Dot([x, y])
        with WaveRunner() as wave_runner:
            # Send the function to the wave runner
            try:
                observed_z = wave_runner.send_function(dot_func)
            except Exception as e:
                self.fail(f"Exception occurred: {e}")
        # observed_z = dot_func.output
        # Print the result
        print("Dot product result:")
        print("Expected:")
        print(expected_z)
        print("Observed:")
        print(observed_z)
        # Check the result
        self.assertTrue(expected_z == observed_z)

    def op_unittest_mul(self):
        # Test multiplication
        # Create the input tensors
        x = TensorU8((2, 2), np.array([[1, 2], [3, 4]]))
        y = TensorU8((2, 2), np.array([[5, 6], [7, 8]]))
        expected_z = TensorU8((2, 2), np.array([[5, 12], [21, 32]]))
        mul_func = Matmul([x, y])
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
