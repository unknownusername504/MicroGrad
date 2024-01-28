import unittest
import numpy as np
from micrograd.functions.primitive_ops import Add
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
            observed_z = wave_runner.send_function(add_func)
        # observed_z = add_func.output
        # Print the result
        print("Addition result:")
        print("Expected:")
        print(expected_z)
        print("Observed:")
        print(observed_z)
        # Check the result
        self.assertTrue(expected_z == observed_z)

    @unittest.skip("Not implemented")
    def op_unittest_sub(self):
        # Test subtraction
        pass

    @unittest.skip("Not implemented")
    def op_unittest_dot(self):
        # Test dot product
        pass

    @unittest.skip("Not implemented")
    def op_unittest_mul(self):
        # Test multiplication
        pass
