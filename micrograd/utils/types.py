from typing import Union, get_args, get_origin
from micrograd.utils.debug_utils import debug_print
from micrograd.tensors.tensor import Tensor, Scalar
import numpy as np
from typing import List

# TODO: Fix this garbage
ScalarLike = Union[np.number, int, float, Scalar]
TensorLike = Union[np.ndarray, List[ScalarLike], Tensor]


def is_scalar_like(value: ScalarLike) -> bool:
    assert get_origin(ScalarLike) is Union
    classes = get_args(ScalarLike)
    debug_print(
        f"ScalarLike: {classes}, type value: {type(value)}, type classes: {type(classes)}"
    )
    return isinstance(value, classes)


def is_tensor_like(value: TensorLike) -> bool:
    assert get_origin(TensorLike) is Union
    classes = get_args(TensorLike)
    debug_print(
        f"TensorLike: {classes}, type value: {type(value)}, type classes: {type(classes)}"
    )
    return isinstance(value, get_args(TensorLike))
