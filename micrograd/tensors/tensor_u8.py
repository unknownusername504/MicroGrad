from typing import ClassVar, List, Optional, Tuple, Union
import numpy as np
from micrograd.tensors.tensor import Tensor


# Class that implements a tensor of unsigned 8-bit integers
# All operations are quantized and performed on the CPU
class TensorU8(Tensor):
    default_dtype: ClassVar[np.dtype] = np.uint8

    # TODO: Implement type bounds checking
    def __init__(
        self,
        value: Union[List, np.ndarray, "Tensor", np.number, int, float] = None,
        shape: Optional[Tuple[int]] = None,
        requires_grad: bool = False,
    ):
        if value is None:
            assert shape is not None
            value = np.zeros(shape, dtype=TensorU8.default_dtype)
        else:
            if not isinstance(value, np.ndarray):
                if type(value) in [int, float, np.number]:
                    value = np.array([value], dtype=TensorU8.default_dtype)
                elif type(value) is List:
                    value = np.array(value, dtype=TensorU8.default_dtype)
                elif isinstance(value, Tensor):
                    value = value.value
                else:
                    raise Exception("Invalid value type")
        if shape is None:
            shape = value.shape
        if value.shape != shape:
            raise Exception("Value shape does not match tensor shape")
        if value.dtype != TensorU8.default_dtype:
            value = value.astype(TensorU8.default_dtype)
        super().__init__(value=value, shape=shape, requires_grad=requires_grad)
