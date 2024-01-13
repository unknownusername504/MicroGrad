import unittest
import numpy as np
from micrograd.tensors.tensor_u8 import TensorU8
from micrograd.functions.wave_process import WaveRunner


# Unit tests of operations
class TestOps(unittest.TestCase):
    def op_unittest_add(self):
        with WaveRunner(debug=True):
            # Test addition
            # Create the input tensors
            x = TensorU8((2, 2), np.array([[1, 2], [3, 4]]))
            y = TensorU8((2, 2), np.array([[5, 6], [7, 8]]))
            # Create the output tensor
            z = x + y
            # Print the result
            print(z.value)

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
