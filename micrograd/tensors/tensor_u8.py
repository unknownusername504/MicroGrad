from typing import List, Union
import numpy as np
from micrograd.tensors.tensor import Tensor


# Class that implements a tensor of unsigned 8-bit integers
# All operations are quantized and performed on the CPU
class TensorU8(Tensor):
    def __init__(
        self,
        shape,
        value: Union[List, np.ndarray, "Tensor", np.number, int, float] = None,
    ):
        if value is None:
            value = np.zeros(shape, dtype=np.uint8)
        else:
            if not isinstance(value, np.ndarray):
                if type(value) in [int, float, np.number]:
                    value = np.array([value], dtype=np.uint8)
                elif type(value) is List:
                    value = np.array(value, dtype=np.uint8)
                elif isinstance(value, Tensor):
                    value = value.value
                else:
                    raise Exception("Invalid value type")
        if value.shape != shape:
            raise Exception("Value shape does not match tensor shape")
        if value.dtype != np.uint8:
            value = value.astype(np.uint8)
        super().__init__(shape=shape, value=value)
