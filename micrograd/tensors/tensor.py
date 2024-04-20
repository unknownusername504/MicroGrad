from typing import List, Optional, Tuple, Union
import numpy as np
from typing import ClassVar

from micrograd.utils.debug_utils import debug_print


class Function:
    def __init__(self, inputs: List["Tensor"]):
        if len(inputs) == 0:
            raise Exception("No inputs provided")
        elif len(inputs) == 1:
            self.input = inputs[0]
        else:
            self.inputs = inputs
        self.output = None

    def _forward(self):
        raise NotImplementedError

    def _backward(self):
        raise NotImplementedError

    def __call__(self):
        output = self._forward()
        # FIXME: Sould I only return tensors?
        if not isinstance(output, Tensor):
            output = Tensor(value=output)
        self.output = output
        if Tensor.auto_grad:
            self._backward()
        return self.output


class Tensor:
    # Class var to turn on/off gradient computation
    # Should only be modified using the with_auto_grad context manager
    auto_grad: ClassVar[bool] = False
    default_dtype: ClassVar[np.dtype] = np.float64

    class with_auto_grad:
        def __init__(self, auto_grad: bool):
            self.auto_grad = auto_grad
            self.restore_auto_grad = Tensor.auto_grad

        def __enter__(self):
            Tensor.auto_grad = self.auto_grad

        def __exit__(self, exc_type, exc_value, traceback):
            Tensor.auto_grad = self.restore_auto_grad

    def __init__(
        self,
        value: Union[List, np.ndarray, "Tensor", np.number, int, float] = None,
        shape: Optional[Tuple[int]] = None,
        requires_grad: bool = True,
    ):
        # If the value is None then create a new numpy array of zeros
        if value is None:
            assert shape is not None
            self.value = np.zeros(shape, dtype=Tensor.default_dtype)
        else:
            if isinstance(value, np.ndarray):
                self.dtype = value.dtype
                self.value = value
            elif type(value) is List:
                # TODO: Infer type from list
                self.dtype = np.float64
                self.value = np.array(value, dtype=self.dtype)
            elif isinstance(value, Tensor):
                self.dtype = value.dtype
                self.value = value.value
            elif isinstance(value, np.number):
                self.dtype = value.dtype
                self.value = np.array([value], dtype=self.dtype)
            elif type(value) is int:
                self.dtype = np.int64
                self.value = np.array([value], dtype=self.dtype)
            elif type(value) is float:
                self.dtype = np.float64
                self.value = np.array([value], dtype=self.dtype)
            else:
                raise Exception("Invalid value type")
        # If the shape is None then set the shape to the value shape
        if shape is None:
            self.shape = self.value.shape
        else:
            self.shape = shape
            self.value = self.value.reshape(self.shape)
        self.requires_grad = requires_grad
        if self.requires_grad:
            self.zero_grad()

    # Function to zero the gradient
    def zero_grad(self):
        self.grad = np.zeros(self.shape, dtype=self.dtype)

    # Length of the tensor
    def __len__(self):
        return len(self.value)

    # Maximum value of the tensor
    def max(self):
        return self.value.max()

    # Casting the tensor dytpe
    def astype(self, dtype):
        self.value = self.value.astype(dtype)
        self.dtype = dtype
        return self

    def tolist(self):
        return self.value.tolist()

    def item(self):
        return self.value.item()

    # Function to test if the tensor is equal to another tensor
    # Only checks the value
    def __eq__(self, other: "Tensor") -> bool:
        return np.array_equal(self.value, other.value)

    # Slice the tensor
    def __getitem__(self, key: Union[int, slice, Tuple[Union[int, slice]]]) -> "Tensor":
        debug_print("self", self)
        debug_print("key:", key)
        num_vals = np.prod(self.shape)
        debug_print("val_len:", num_vals)
        # Make sure the key range is valid
        if type(key) is tuple:
            indices = np.ravel_multi_index(key, self.shape)
            if isinstance(indices, np.ndarray):
                for index in indices:
                    if (index < 0) or (index > num_vals):
                        raise Exception("Invalid key range")
            elif isinstance(indices, np.number):
                if (indices < 0) or (indices > num_vals):
                    raise Exception("Invalid key range")
            else:
                debug_print("indices:", indices)
                debug_print("type(indices):", type(indices))
                raise Exception("Invalid indices type")
        if type(key) is int:
            if (key < 0) or (key > num_vals):
                raise Exception("Invalid key range")
        if type(key) is slice:
            if (key.start < 0) or (key.stop > num_vals):
                raise Exception("Invalid key range")
        tensor_type = type(self)
        debug_print("getting value_slice")
        # Get the value
        value_slice = self.value[key]
        debug_print("value_slice:", value_slice)
        if not isinstance(value_slice, np.ndarray):
            debug_print("value_slice is not np.ndarray, must be a scalar")
            value_slice = np.array([value_slice], dtype=self.dtype)
        debug_print("value_slice:", value_slice)
        # Length of the slice
        length = len(value_slice)
        # Shape is now flattened
        shape_slice = (length,)
        # Create the tensor
        tensor_slice = tensor_type(shape=shape_slice, value=value_slice)
        # Return the tensor
        return tensor_slice

    # Set slice of the tensor
    def __setitem__(
        self,
        key: Union[int, slice, Tuple[Union[int, slice]]],
        value: Union[List, np.ndarray, "Tensor", np.number, int, float],
    ):
        debug_print("key:", key)
        num_vals = np.prod(self.shape)
        num_indices = 0
        debug_print("val_len:", num_vals)
        # Make sure the key range is valid
        if type(key) is tuple:
            debug_print("key is tuple")
            if len(key) == 0:
                raise Exception("Invalid key range")
            if len(key) == 1:
                # If the key is a single value then convert it to a np.number
                indices = key[0]
            else:
                indices = np.ravel_multi_index(key, self.shape)
            debug_print("indices:", indices)
            if isinstance(indices, np.ndarray):
                num_indices = len(indices)
                for index in indices:
                    if (index < 0) or (index > num_vals):
                        raise Exception("Invalid key range")
            elif isinstance(indices, np.number):
                num_indices = 1
                if (indices < 0) or (indices > num_vals):
                    raise Exception("Invalid key range")
            else:
                debug_print("indices:", indices)
                debug_print("type(indices):", type(indices))
                raise Exception("Invalid indices type")
        if type(key) is int:
            debug_print("key is int")
            num_indices = 1
            if (key < 0) or (key > num_vals):
                raise Exception("Invalid key range")
        if type(key) is slice:
            debug_print("key is slice")
            num_indices = (key.stop - key.start) // key.step
            if (key.start < 0) or (key.stop > num_vals):
                raise Exception("Invalid key range")
        debug_print("made it past key range")
        if num_indices == 0:
            raise Exception("Invalid key range")
        debug_print("value:", value)
        if isinstance(value, Tensor):
            value = value.value
        elif type(value) is List:
            debug_print("value is list")
            value = np.array(value)
        elif type(value) in [np.number, int, float]:
            debug_print("value is scalar")
            value = np.array([value])
        debug_print("value:", value)
        if num_indices != len(value):
            raise Exception("Invalid value length")
        # Flatten the incoming tensor value and ensure it is a numpy array of lesser or equal size
        value = value.flatten()
        # Ensure that the value is compatible with the tensor dtype with an attempt to cast
        try:
            value = value.astype(self.dtype)
        except:
            raise Exception("Incompatible value dtype")
        # Set the value
        debug_print("setting value")
        self.value[key] = value
        debug_print("set value")

    # Print string representation of the tensor
    def __str__(self):
        return f"Tensor(shape={self.shape}, dtype={self.dtype}, value={self.value})"

    # Print representation of the tensor
    def __repr__(self):
        return self.__str__()

    # Get the shape of the tensor
    def get_shape(self):
        return self.shape

    # Get the dtype of the tensor
    def get_dtype(self):
        return self.dtype

    # Get the value of the tensor
    def get_value(self):
        return self.value

    # Flatten the tensor
    def flatten(self):
        self.value = self.value.flatten()
        self.shape = (len(self.value),)
        return self

    # Reshape the tensor
    def reshape(self, shape: Tuple[int]):
        self.value = self.value.reshape(shape)
        self.shape = shape
        return self

    # Transpose the tensor
    def transpose(self, axes: Optional[Tuple[int]] = None):
        self.value = self.value.transpose(axes)
        self.shape = self.value.shape
        return self

    def expand_dims(self, axis: Union[int, Tuple[int]]):
        self.value = np.expand_dims(self.value, axis)
        self.shape = self.value.shape
        return self

    # Join a sequence of arrays along an existing axis.
    def concatenate(self, tensors, axis=0):
        tensors = [tensor.value for tensor in tensors]
        self.value = np.concatenate([self.value] + tensors, axis=axis)
        self.shape = self.value.shape
        return self

    @staticmethod
    def get_output_shape(x: "Tensor", y: "Tensor") -> Tuple[int, ...]:
        x_shape, y_shape = list(x.shape), list(y.shape)
        if x_shape[0] == y_shape[0]:
            expand_dir = "right"
        elif x_shape[-1] == y_shape[-1]:
            expand_dir = "left"
        else:
            raise Exception(
                "Cannot broadcast tensors of different shapes x:{} y:{}".format(
                    x_shape, y_shape
                )
            )

        # Pad the smaller shape with 1s
        if len(x_shape) < len(y_shape):
            if expand_dir == "right":
                x_shape = x_shape + ([1] * (len(y_shape) - len(x_shape)))
            elif expand_dir == "left":
                x_shape = ([1] * (len(y_shape) - len(x_shape))) + x_shape
        elif len(x_shape) > len(y_shape):
            if expand_dir == "right":
                y_shape = y_shape + ([1] * (len(x_shape) - len(y_shape)))
            elif expand_dir == "left":
                y_shape = ([1] * (len(x_shape) - len(y_shape))) + y_shape

        assert len(x_shape) == len(y_shape)

        debug_print("x_shape:{} y_shape:{}".format(x_shape, y_shape))
        output_shape = []

        # Check if the shapes are compatible
        for dim1, dim2 in zip(x_shape, y_shape):
            if dim1 != dim2 and dim1 != 1 and dim2 != 1:
                raise Exception(
                    "Cannot broadcast tensors of different shapes x:{} y:{}".format(
                        x_shape, y_shape
                    )
                )
            else:
                output_shape.append(max(dim1, dim2))

        debug_print("output_shape:", output_shape)

        return tuple(output_shape)

    @staticmethod
    def broadcast(x: "Tensor", y: "Tensor") -> Tuple["Tensor", "Tensor"]:
        # We want to broadcast the tensors
        # Get the number of dimensions
        output_shape = Tensor.get_output_shape(x, y)
        # Create the output tensors
        x_out = x.reshape(output_shape)
        y_out = y.reshape(output_shape)
        # Return the output tensors
        return x_out, y_out

    @staticmethod
    def get_output_data_type(x: "Tensor", y: "Tensor") -> np.dtype:
        # We want to determine the output type based on the input types
        # If the tensor types are not the same then take the higher type
        # List of known data types in order of precedence
        # Get all numpy datatypes
        known_data_types = np.sctypes["float"] + np.sctypes["int"] + np.sctypes["uint"]
        data_type_1 = x.dtype
        data_type_2 = y.dtype
        for data_type in known_data_types:
            if data_type_1 == data_type:
                return data_type
            if data_type_2 == data_type:
                return data_type
        raise Exception("Unknown data type")

    @staticmethod
    def get_output_tensor_type(x: "Tensor", y: "Tensor") -> type:
        # We want to determine the output type based on the input types
        # If the tensor types are not the same then take the higher type
        # List of known data types in order of precedence
        known_data_types = np.sctypes["float"] + np.sctypes["int"] + np.sctypes["uint"]
        data_type_1 = x.dtype
        data_type_2 = y.dtype
        for data_type in known_data_types:
            if data_type_1 == data_type:
                return type(x)
            if data_type_2 == data_type:
                return type(y)
        raise Exception("Unknown tensor type")

    class Add(Function):
        def __init__(self, inputs):
            super().__init__(inputs)
            x = self.inputs[0]
            y = self.inputs[1]
            output_tensor_type = Tensor.get_output_tensor_type(x, y)
            output_dtype = Tensor.get_output_data_type(x, y)
            output_shape = Tensor.get_output_shape(x, y)
            self.output = output_tensor_type(np.zeros(output_shape, dtype=output_dtype))

        # Should not be called directly, prefer to use the __call__ method directly or indirectly
        def _forward(self):
            self.output.value = Tensor.Add.add(
                self.inputs[0].get_value(), self.inputs[1].get_value()
            )
            return self.output

        # Should not be called directly, prefer to use the __call__ method directly or indirectly
        # Should only be called with auto_grad=True
        def _backward(self):
            if Tensor.auto_grad:
                self.inputs[0].grad = Tensor.Add.add(
                    self.inputs[0].grad, self.output.grad
                )
                self.inputs[1].grad = Tensor.Add.add(
                    self.inputs[1].grad, self.output.grad
                )
            else:
                raise Exception("Backward should only be called with auto_grad=True")

        @staticmethod
        def add(
            x: Union["Tensor", np.ndarray], y: Union["Tensor", np.ndarray]
        ) -> np.ndarray:
            debug_print("add x:", x)
            debug_print("add y:", y)
            if isinstance(x, Tensor):
                x = x.get_value()
            if isinstance(y, Tensor):
                y = y.get_value()
            # Perform the addition
            output = x + y
            if not isinstance(output, np.ndarray):
                # Must be a scalar
                output = np.array([output])
            debug_print("output:", output)
            return output

    class Sub(Function):
        def __init__(self, inputs):
            super().__init__(inputs)
            x = self.inputs[0]
            y = self.inputs[1]
            output_tensor_type = Tensor.get_output_tensor_type(x, y)
            output_dtype = Tensor.get_output_data_type(x, y)
            output_shape = Tensor.get_output_shape(x, y)
            self.output = output_tensor_type(np.zeros(output_shape, dtype=output_dtype))

        # Should not be called directly, prefer to use the __call__ method directly or indirectly
        def _forward(self):
            self.output.value = Tensor.Sub.sub(
                self.inputs[0].get_value(), self.inputs[1].get_value()
            )
            return self.output

        # Should not be called directly, prefer to use the __call__ method directly or indirectly
        # Should only be called with auto_grad=True
        def _backward(self):
            if Tensor.auto_grad:
                self.inputs[0].grad = Tensor.Add.add(
                    self.inputs[0].grad, self.output.grad
                )
                self.inputs[1].grad = Tensor.Sub.sub(
                    self.inputs[1].grad, self.output.grad
                )
            else:
                raise Exception("Backward should only be called with auto_grad=True")

        @staticmethod
        def sub(
            x: Union["Tensor", np.ndarray], y: Union["Tensor", np.ndarray]
        ) -> np.ndarray:
            debug_print("sub x:", x)
            debug_print("sub y:", y)
            if isinstance(x, Tensor):
                x = x.get_value()
            if isinstance(y, Tensor):
                y = y.get_value()
            # Perform the subtraction
            output = x - y
            if not isinstance(output, np.ndarray):
                # Must be a scalar
                output = np.array([output])
            debug_print("output:", output)
            return output

    class Dot(Function):
        def __init__(self, inputs):
            super().__init__(inputs)
            x = self.inputs[0]
            y = self.inputs[1]
            output_tensor_type = Tensor.get_output_tensor_type(x, y)
            output_dtype = Tensor.get_output_data_type(x, y)
            output_shape = Tensor.get_output_shape(x, y)
            self.output = output_tensor_type(np.zeros(output_shape, dtype=output_dtype))

        # Should not be called directly, prefer to use the __call__ method directly or indirectly
        def _forward(self):
            self.output.value = self.dot(
                self.inputs[0].get_value(), self.inputs[1].get_value()
            )
            return self.output

        # Should not be called directly, prefer to use the __call__ method directly or indirectly
        # Should only be called with auto_grad=True
        def _backward(self):
            if Tensor.auto_grad:
                self.inputs[0].grad = Tensor.Add.add(
                    self.inputs[0].grad,
                    self.dot(self.inputs[1].get_value(), self.output.grad),
                )
                self.inputs[1].grad = Tensor.Add.add(
                    self.inputs[1].grad,
                    self.dot(self.inputs[0].get_value(), self.output.grad),
                )
            else:
                raise Exception("Backward should only be called with auto_grad=True")

        @staticmethod
        def dot(
            x: Union["Tensor", np.ndarray], y: Union["Tensor", np.ndarray]
        ) -> np.ndarray:
            debug_print("dot x:", x)
            debug_print("dot y:", y)
            if isinstance(x, Tensor):
                x = x.get_value()
            if isinstance(y, Tensor):
                y = y.get_value()
            # Perform the dot product
            output = np.dot(x, y)
            if not isinstance(output, np.ndarray):
                # Must be a scalar
                output = np.array([output])
            debug_print("output:", output)
            return output

    class Mul(Function):
        def __init__(self, inputs):
            super().__init__(inputs)
            x = self.inputs[0]
            y = self.inputs[1]
            output_tensor_type = Tensor.get_output_tensor_type(x, y)
            output_dtype = Tensor.get_output_data_type(x, y)
            output_shape = Tensor.get_output_shape(x, y)
            self.output = output_tensor_type(np.zeros(output_shape, dtype=output_dtype))

        # Should not be called directly, prefer to use the __call__ method directly or indirectly
        def _forward(self):
            self.output.value = Tensor.Mul.mul(
                self.inputs[0].get_value(), self.inputs[1].get_value()
            )
            return self.output

        # Should not be called directly, prefer to use the __call__ method directly or indirectly
        # Should only be called with auto_grad=True
        def _backward(self):
            if Tensor.auto_grad:
                self.inputs[0].grad = Tensor.Add.add(
                    self.inputs[0].grad, self.inputs[1].get_value() * self.output.grad
                )
                self.inputs[1].grad = Tensor.Add.add(
                    self.inputs[1].grad, self.inputs[0].get_value() * self.output.grad
                )
            else:
                raise Exception("Backward should only be called with auto_grad=True")

        @staticmethod
        def mul(
            x: Union["Tensor", np.ndarray], y: Union["Tensor", np.ndarray]
        ) -> np.ndarray:
            debug_print("mul x:", x)
            debug_print("mul y:", y)
            if isinstance(x, Tensor):
                x = x.get_value()
            if isinstance(y, Tensor):
                y = y.get_value()
            # Perform the multiplication
            output = x * y
            if not isinstance(output, np.ndarray):
                # Must be a scalar
                output = np.array([output])
            debug_print("output:", output)
            return output

    class Matmul(Function):
        def __init__(self, inputs):
            super().__init__(inputs)
            x = self.inputs[0]
            y = self.inputs[1]
            output_tensor_type = Tensor.get_output_tensor_type(x, y)
            output_dtype = Tensor.get_output_data_type(x, y)
            output_shape = Tensor.get_output_shape(x, y)
            self.output = output_tensor_type(np.zeros(output_shape, dtype=output_dtype))

        # Should not be called directly, prefer to use the __call__ method directly or indirectly
        def _forward(self):
            self.output.value = Tensor.Matmul.matmul(
                self.inputs[0].get_value(), self.inputs[1].get_value()
            )
            return self.output

        # Should not be called directly, prefer to use the __call__ method directly or indirectly
        # Should only be called with auto_grad=True
        def _backward(self):
            if Tensor.auto_grad:
                self.inputs[0].grad = Tensor.Add.add(
                    self.inputs[0].grad,
                    Tensor.Matmul.matmul(
                        self.output.grad, self.inputs[1].get_value().T
                    ),
                )
                self.inputs[1].grad = Tensor.Add.add(
                    self.inputs[1].grad,
                    Tensor.Matmul.matmul(
                        self.inputs[0].get_value().T, self.output.grad
                    ),
                )
            else:
                raise Exception("Backward should only be called with auto_grad=True")

        @staticmethod
        def matmul(
            x: Union["Tensor", np.ndarray], y: Union["Tensor", np.ndarray]
        ) -> np.ndarray:
            debug_print("matmul x:", x)
            debug_print("matmul y:", y)
            if isinstance(x, Tensor):
                x = x.get_value()
            if isinstance(y, Tensor):
                y = y.get_value()
            # Perform the matrix multiplication
            print("x.shape:", x.shape)
            print("y.shape:", y.shape)
            output = x @ y
            if not isinstance(output, np.ndarray):
                # Must be a scalar
                output = np.array([output])
            debug_print("output:", output)
            return output

    class Div(Function):
        def __init__(self, inputs):
            super().__init__(inputs)
            # Set the gradient function
            x = self.inputs[0]
            y = self.inputs[1]
            output_tensor_type = Tensor.get_output_tensor_type(x, y)
            output_dtype = Tensor.get_output_data_type(x, y)
            output_shape = Tensor.get_output_shape(x, y)
            self.output = output_tensor_type(np.zeros(output_shape, dtype=output_dtype))

        # Should not be called directly, prefer to use the __call__ method directly or indirectly
        def _forward(self):
            self.output.value = Tensor.Div.div(
                self.inputs[0].get_value(), self.inputs[1].get_value()
            )
            return self.output

        # Should not be called directly, prefer to use the __call__ method directly or indirectly
        # Should only be called with auto_grad=True
        def _backward(self):
            if Tensor.auto_grad:
                self.inputs[0].grad = Tensor.Add.add(
                    self.inputs[0].grad,
                    Tensor.Div.div(self.output.grad, self.inputs[1].get_value()),
                )
                self.inputs[1].grad = Tensor.Add.add(
                    self.inputs[1].grad,
                    Tensor.Div.div(
                        self.inputs[0].get_value(), self.inputs[1].get_value()
                    )
                    * -1
                    * self.output.grad,
                )
            else:
                raise Exception("Backward should only be called with auto_grad=True")

        @staticmethod
        def div(
            x: Union["Tensor", np.ndarray], y: Union["Tensor", np.ndarray]
        ) -> np.ndarray:
            debug_print("div x:", x)
            debug_print("div y:", y)
            if isinstance(x, Tensor):
                x = x.get_value()
            if isinstance(y, Tensor):
                y = y.get_value()
            # Perform the division
            output = x / y
            if not isinstance(output, np.ndarray):
                # Must be a scalar
                output = np.array([output])
            debug_print("output:", output)
            return output

    class Neg(Function):
        def __init__(self, inputs):
            super().__init__(inputs)
            x = self.inputs[0]
            output_tensor_type = type(x)
            output_dtype = output_tensor_type.default_dtype
            output_shape = x.shape
            self.output = output_tensor_type(np.zeros(output_shape, dtype=output_dtype))

        # Should not be called directly, prefer to use the __call__ method directly or indirectly
        def _forward(self):
            self.output.value = Tensor.Neg.neg(self.inputs[0].get_value())
            return self.output

        # Should not be called directly, prefer to use the __call__ method directly or indirectly
        # Should only be called with auto_grad=True
        def _backward(self):
            if Tensor.auto_grad:
                self.inputs[0].grad = Tensor.Neg.neg(self.output.grad)
            else:
                raise Exception("Backward should only be called with auto_grad=True")

        @staticmethod
        def neg(x: Union["Tensor", np.ndarray]) -> np.ndarray:
            debug_print("neg x:", x)
            if isinstance(x, Tensor):
                x = x.get_value()
            # Perform the negation
            output = -x
            if not isinstance(output, np.ndarray):
                # Must be a scalar
                output = np.array([output])
            debug_print("output:", output)
            return output

    def __add__(self, other: Union["Tensor", np.ndarray, int, float]) -> "Tensor":
        return Tensor.Add([self, other])()

    def __radd__(self, other: Union["Tensor", np.ndarray, int, float]) -> "Tensor":
        return other + self

    def __sub__(self, other: Union["Tensor", np.ndarray, int, float]) -> "Tensor":
        return Tensor.Sub([self, other])()

    def __rsub__(self, other: Union["Tensor", np.ndarray, int, float]) -> "Tensor":
        return other - self

    def __mul__(self, other: Union["Tensor", np.ndarray, int, float]) -> "Tensor":
        return Tensor.Mul([self, other])()

    def __rmul__(self, other: Union["Tensor", np.ndarray, int, float]) -> "Tensor":
        return other * self

    def __truediv__(self, other: Union["Tensor", np.ndarray, int, float]) -> "Tensor":
        return Tensor.Div([self, other])()

    def __rtruediv__(self, other: Union["Tensor", np.ndarray, int, float]) -> "Tensor":
        return other / self

    def __matmul__(self, other: Union["Tensor", np.ndarray, int, float]) -> "Tensor":
        # Perform the dot product for vectors and matrix multiplication for matrices
        if len(self.shape) == 1 and len(other.shape) == 1:
            return Tensor.Dot([self, other])()
        else:
            return Tensor.Matmul([self, other])()

    def __rmatmul__(self, other: Union["Tensor", np.ndarray, int, float]) -> "Tensor":
        return other @ self

    def __neg__(self) -> "Tensor":
        return Tensor.Neg([self])()


class Scalar(Tensor):
    class InvalidScalarOperation(Exception):
        # Takes the function name and exception is "Cannot __func__ a scalar"
        def __init__(self, func_name: str):
            super().__init__(f"Cannot {func_name} a scalar")

    def __init__(self, value: Union[np.number, int, float]):
        super().__init__(value=value, shape=(1,))

        # Undefine any operations that would reshape the tensor to become non-scalar
        methods_to_undefine = [
            "reshape",
            "expand_dims",
            "concatenate",
            "transpose",
            "__getitem__",
            "__setitem__",
        ]

        # Undefine the methods
        for method_name in methods_to_undefine:

            def invalid_operation(*args, **kwargs):
                self._raise_invalid_operation(method_name)

            setattr(self, method_name, invalid_operation)

    def _raise_invalid_operation(self, method_name):
        raise Scalar.InvalidScalarOperation(method_name)
