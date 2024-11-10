from typing import List, Optional, Tuple, Union, ClassVar
import numpy as np

from micrograd.utils.debug_utils import debug_print

# TODO: Fix this garbage
ScalarLikeForward = Union[np.number, int, float, "Scalar"]
TensorLikeForward = Union[np.ndarray, List[ScalarLikeForward], "Tensor"]


class Function:
    def __init__(self, inputs: List["Tensor"], function: str):
        if len(inputs) == 0:
            raise Exception("No inputs provided")
        self.inputs = inputs
        if len(inputs) == 1:
            self.input = inputs[0]
        self.inputs_requires_grad = any([input.requires_grad for input in inputs])

        if function != None:
            x = self.inputs[0]
            y = self.inputs[1] if len(self.inputs) > 1 else None
            output_tensor_type = Tensor.get_output_tensor_type(x, y, function=function)
            output_dtype = Tensor.get_output_data_type(x, y, function=function)
            output_shape = Tensor.get_output_shape(x, y, function=function)
            if output_tensor_type == Scalar:
                is_float = np.issubdtype(output_dtype, np.floating)
                self.output = output_tensor_type(
                    0.0 if is_float else 0,
                    requires_grad=self.inputs_requires_grad,
                )
            else:
                self.output = output_tensor_type(
                    np.empty(shape=output_shape, dtype=output_dtype),
                    requires_grad=self.inputs_requires_grad,
                )
        else:
            self.output = None

    def _forward(self):
        raise NotImplementedError

    def _backward(self):
        raise NotImplementedError

    def __call__(self):
        # This will ensure all operations for this function are done in the correct autograd context,
        # Even if the function is called in a different context/proc
        self._forward()
        if Tensor.get_auto_grad() and self.inputs_requires_grad:
            self._backward()
        return self.output


class Tensor:
    # Class var to turn on/off gradient computation
    # Should only be modified using the with_auto_grad context manager
    auto_grad: ClassVar[bool] = False
    default_dtype: ClassVar[np.dtype] = np.float64

    @staticmethod
    def get_auto_grad():
        return Tensor.auto_grad

    @staticmethod
    def set_auto_grad(update_auto_grad: bool):
        Tensor.auto_grad = update_auto_grad

    class with_auto_grad:
        def __init__(self, update_auto_grad: bool):
            self.update_auto_grad = update_auto_grad
            self.restore_auto_grad = Tensor.get_auto_grad()

        def __enter__(self):
            Tensor.set_auto_grad(self.update_auto_grad)

        def __exit__(self, exc_type, exc_value, traceback):
            Tensor.set_auto_grad(self.restore_auto_grad)

    def __init__(
        self,
        value: Union[TensorLikeForward, ScalarLikeForward] = None,
        shape: Optional[Tuple[int]] = None,
        requires_grad: bool = False,
    ):
        if value is None:
            assert shape is not None
            self._create_tensor(shape)
        else:
            if isinstance(value, np.ndarray):
                self.value = value
            elif isinstance(value, list) or isinstance(value, List):
                # Infer type and create array
                arr = np.array(value, dtype=Tensor.default_dtype)
                self.value = arr
            elif isinstance(value, Tensor):
                self.value = value.value
            elif isinstance(value, np.number):
                self.value = np.array([value], dtype=value.dtype)
            elif isinstance(value, int):
                self.value = np.array([value], dtype=np.int64)
            elif isinstance(value, float):
                self.value = np.array([value], dtype=np.float64)
            else:
                # Handle invalid types
                raise Exception(f"Invalid value type: {type(value)}")

        # Set shape if provided
        if shape is not None:
            self.value = self.value.reshape(shape)

        self.requires_grad = requires_grad
        if self.requires_grad:
            self.zero_grad()

    def _create_tensor(self, shape):
        self.value = np.empty(shape=shape, dtype=Tensor.default_dtype)

    def zero_grad(self):
        if self.requires_grad:
            self.grad = np.zeros_like(self.value, dtype=Tensor.default_dtype)

    # Length of the tensor
    def __len__(self):
        if len(self.value.shape) == 0:
            return 1
        return len(self.value)

    # Size of the tensor
    def get_size(self):
        return self.value.size

    @property
    def size(self):
        return self.get_size()

    # Maximum value of the tensor
    def max(self):
        return self.value.max()

    def argmax(self, axis=None):
        return self.value.argmax(axis=axis)

    # Casting the tensor dytpe
    def astype(self, dtype):
        self.value = self.value.astype(dtype)
        if self.requires_grad:
            self.grad = self.grad.astype(dtype)
        return self

    def tolist(self):
        return self.value.tolist()

    def item(self):
        return self.value.item()

    def reduce(self, axis=None, keepdims=False, reduction="mean"):
        if reduction not in ["mean", "sum"]:
            raise ValueError(
                f"Invalid reduction type: {reduction}. Supported: 'mean', 'sum'."
            )

        # Apply reduction to the tensor value
        if reduction == "mean":
            self.value = np.mean(self.value, axis=axis, keepdims=keepdims).astype(
                self.value.dtype
            )
            if self.requires_grad:
                # For mean reduction, scale gradients by the size of the reduced axis to keep gradients aligned
                grad_scale = np.prod(self.value.shape) / np.prod(self.grad.shape)
                self.grad = (
                    np.ones_like(
                        self.value,
                        dtype=self.grad.dtype,
                    )
                    * grad_scale
                )
        elif reduction == "sum":
            self.value = np.sum(self.value, axis=axis, keepdims=keepdims).astype(
                self.value.dtype
            )
            if self.requires_grad:
                self.grad = np.ones_like(
                    self.value,
                    dtype=self.grad.dtype,
                )

    # Function to test if the tensor is equal to another tensor
    # Only checks the value
    def __eq__(self, other: "Tensor") -> bool:
        return np.array_equal(self.value, other.value)

    # Slice the tensor
    def __getitem__(self, key: Union[int, slice, Tuple[Union[int, slice]]]) -> "Tensor":
        # TODO: Use viewable tensors
        return Tensor(self.value[key], requires_grad=self.requires_grad)

    # Set slice of the tensor
    def __setitem__(
        self,
        key: Union[int, slice, Tuple[Union[int, slice]]],
        value: Union[List, np.ndarray, "Tensor", np.number, int, float],
    ):
        # We cannot set the value of a tensor that requires grad
        if self.requires_grad:
            raise Exception("Cannot set value of tensor that requires grad")
        self.value[key] = value

    # Print string representation of the tensor
    def __str__(self):
        return f"Tensor(shape={self.shape}, dtype={self.dtype}, value={self.value})"

    # Print representation of the tensor
    def __repr__(self):
        return self.__str__()

    # Get the shape of the tensor
    def get_shape(self):
        return self.value.shape

    @property
    def shape(self):
        return self.get_shape()

    def get_ndim(self):
        return self.value.ndim

    @property
    def ndim(self):
        return self.get_ndim()

    # Get the dtype of the tensor
    def get_dtype(self):
        return self.value.dtype

    @property
    def dtype(self):
        return self.get_dtype()

    # Get the value of the tensor
    def get_value(self):
        return self.value

    @staticmethod
    def is_flat(x: Union["Tensor", np.ndarray]):
        value = x if isinstance(x, np.ndarray) else x.get_value()
        # Check if the tensor is 1D or has only one non-singleton dimension
        return len(value.shape) <= 1 or (
            len(value.shape) > 1 and all(dim == 1 for dim in value.shape[:-1])
        )

    # Flatten the tensor
    def flatten(self):
        self.value = self.value.flatten()
        if self.requires_grad:
            self.grad = self.grad.flatten()
        return self

    # Reshape the tensor
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        self.value = self.value.reshape(shape)
        if self.requires_grad:
            self.grad = self.grad.reshape(shape)
        return self

    # Transpose the tensor
    def transpose(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], tuple):
            axes = axes[0]
        self.value = self.value.transpose(axes)
        if self.requires_grad:
            self.grad = self.grad.transpose(axes)
        return self

    def expand_dims(self, axis: Union[int, Tuple[int]]):
        self.value = np.expand_dims(self.value, axis)
        if self.requires_grad:
            self.grad = np.expand_dims(self.grad, axis)
        return self

    # Join a sequence of arrays along an existing axis.
    def concatenate(self, tensors, axis=0):
        tensors = [tensor.value for tensor in tensors]
        self.value = np.concatenate([self.value] + tensors, axis=axis)
        if self.requires_grad:
            tensor_grads = [tensor.grad for tensor in tensors]
            self.grad = np.concatenate([self.grad] + tensor_grads, axis=axis)
        return self

    @staticmethod
    def split(tensor, indices_or_sections, axis=0):
        # TODO: Use viewable tensors
        requires_grad = tensor.requires_grad
        split_vals = np.split(tensor.value, indices_or_sections, axis=axis)
        return [
            Tensor(value=split_val, requires_grad=requires_grad)
            for split_val in split_vals
        ]

    @staticmethod
    def broadcast_shapes(x_shape: Tuple[int], y_shape: Tuple[int]) -> Tuple[int]:
        return np.broadcast_shapes(x_shape, y_shape)

    @staticmethod
    def calculate_matmul_shape(x_shape: Tuple[int], y_shape: Tuple[int]) -> Tuple[int]:
        # Determine the maximum dimensionality
        assert (
            x_shape[-1] == y_shape[0]
        ), "Incompatible shapes for matmul with shapes {} and {}".format(
            x_shape, y_shape
        )

        # Calculate the broadcasted shape element-wise
        return tuple(x_shape[:-1] + y_shape[1:])

    @staticmethod
    def get_output_shape(x: "Tensor", y: "Tensor", function: str) -> Tuple[int]:
        if y is None:
            return x.shape
        elif x is None:
            return y.shape
        if function == "Matmul":
            return Tensor.calculate_matmul_shape(x.shape, y.shape)
        elif function == "Dot":
            # Assert that x and y are 1D arrays
            assert Tensor.is_flat(x) and Tensor.is_flat(
                y
            ), f"x and y must be 1D arrays: {x.shape}, {y.shape}"
            assert len(x) == len(
                y
            ), f"x and y must have the same length: {len(x)}, {len(y)}"
            return ()
        elif function == "MeanSquaredError":
            assert (
                x.shape == y.shape
            ), f"Incompatible shapes for MSE: {x.shape}, {y.shape}"
            return ()
        elif function == "NegativeLogLikelihood":
            assert Tensor.is_flat(y), f"y must be 1D array: {y.shape}"
            return y.shape
        else:
            return Tensor.broadcast_shapes(x.shape, y.shape)

    @staticmethod
    def get_output_data_type(x: "Tensor", y: "Tensor", function: str) -> np.dtype:
        # We want to determine the output type based on the input types
        # If the tensor types are not the same then take the higher type
        # List of known data types in order of precedence
        # Get all numpy datatypes
        if function in [
            "Div",
            "Sigmoid",
            "Softmax",
            "MeanSquaredError",
            "NegativeLogLikelihood",
        ]:
            # Any fuctions that require division should return a float
            return Tensor.default_dtype
        elif y is None:
            return x.dtype
        elif x is None:
            return y.dtype
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
    def get_output_tensor_type(x: "Tensor", y: "Tensor", function: str) -> type:
        # We want to determine the output type based on the input types
        # If the tensor types are not the same then take the higher type
        # List of known data types in order of precedence
        if function in ["Dot", "MeanSquaredError"]:
            return Scalar
        elif function in ["Div", "Sigmoid", "Softmax"]:
            # Any fuctions that require division should return a float
            return Tensor
        elif y is None:
            return type(x)
        elif x is None:
            return type(y)
        known_data_types = np.sctypes["float"] + np.sctypes["int"] + np.sctypes["uint"]
        data_type_1 = x.dtype
        data_type_2 = y.dtype
        for data_type in known_data_types:
            if data_type_1 == data_type:
                return type(x)
            if data_type_2 == data_type:
                return type(y)
        raise Exception("Unknown tensor type")

        """
        Summary Table of Operations
        Operation | ∂L/∂output | ∂L/∂x | ∂L/∂y
        Add       | 1          | 1     | 1
        Sub       | 1          | 1     | -1
        Dot       | 1          | y     | x
        Mul       | y          | y     | x
        Matmul    | 1          | y.T   | x.T
        Div       | 1/y        | 1/y   | -x/(y^2)
        Neg       | -1         | -1    | N/A
        """

    class Add(Function):
        def __init__(self, inputs):
            super().__init__(inputs=inputs, function=type(self).__name__)

        # Should not be called directly, prefer to use the __call__ method directly or indirectly
        def _forward(self):
            self.output.value = Tensor.Add.add(
                self.inputs[0].get_value(), self.inputs[1].get_value()
            )

        # Should not be called directly, prefer to use the __call__ method directly or indirectly
        # Should only be called with auto_grad=True
        def _backward(self):
            if Tensor.get_auto_grad() and self.inputs_requires_grad:
                # ∂L/∂output = 1
                self.output.grad = np.ones_like(
                    self.output.grad, dtype=Tensor.default_dtype
                )
                # ∂L/∂x = ∂L/∂output * 1
                if self.inputs[0].requires_grad:
                    self.inputs[0].grad = Tensor.Add.add(
                        self.inputs[0].grad, self.output.grad
                    )
                # ∂L/∂y = ∂L/∂output * 1
                if self.inputs[1].requires_grad:
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
            super().__init__(inputs=inputs, function=type(self).__name__)

        # Should not be called directly, prefer to use the __call__ method directly or indirectly
        def _forward(self):
            self.output.value = Tensor.Sub.sub(
                self.inputs[0].get_value(), self.inputs[1].get_value()
            )

        # Should not be called directly, prefer to use the __call__ method directly or indirectly
        # Should only be called with auto_grad=True
        def _backward(self):
            if Tensor.get_auto_grad() and self.inputs_requires_grad:
                # ∂L/∂output = 1
                self.output.grad = np.ones_like(
                    self.output.grad, dtype=Tensor.default_dtype
                )
                # ∂L/∂x = ∂L/∂output * 1
                if self.inputs[0].requires_grad:
                    self.inputs[0].grad = Tensor.Add.add(
                        self.inputs[0].grad, self.output.grad
                    )
                # ∂L/∂y = ∂L/∂output * -1
                if self.inputs[1].requires_grad:
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
            super().__init__(inputs, function=type(self).__name__)

        # Should not be called directly, prefer to use the __call__ method directly or indirectly
        def _forward(self):
            self.output.value = self.dot(
                self.inputs[0].get_value(), self.inputs[1].get_value()
            )

        # Should not be called directly, prefer to use the __call__ method directly or indirectly
        # Should only be called with auto_grad=True
        def _backward(self):
            if Tensor.get_auto_grad() and self.inputs_requires_grad:
                # ∂L/∂output = 1
                self.output.grad = np.ones_like(
                    self.output.grad, dtype=Tensor.default_dtype
                )
                # ∂L/∂x = ∂L/∂output * y
                if self.inputs[0].requires_grad:
                    # self.inputs[0].grad = self.inputs[0].grad + (
                    #    self.inputs[1].get_value() * self.output.grad
                    # )
                    # Skip the mult by grad, since it is a scalar and it is 1 so will not change the value
                    self.inputs[0].grad = (
                        self.inputs[0].grad + self.inputs[1].get_value()
                    )
                # ∂L/∂y = ∂L/∂output * x
                if self.inputs[1].requires_grad:
                    # self.inputs[1].grad = self.inputs[1].grad + (
                    #    self.inputs[0].get_value() * self.output.grad
                    # )
                    # Skip the mult by grad, since it is a scalar (incompat shape possibly in dot) and it is 1 so will not change the value
                    self.inputs[1].grad = (
                        self.inputs[1].grad + self.inputs[0].get_value()
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
            # Assert that x and y are 1D arrays
            assert Tensor.is_flat(x) and Tensor.is_flat(
                y
            ), f"x and y must be 1D arrays: {x.shape}, {y.shape}"
            # assert len(x) == len(
            #    y
            # ), f"x and y must have the same length: {len(x)}, {len(y)}"
            # Perform the dot product
            output = np.dot(x, y)
            if isinstance(output, np.ndarray):
                # Must be a scalar
                output = output.item()
            else:
                # Assert that the output is a scalar
                assert isinstance(output, np.number)
            debug_print("output:", output)
            return output

    class Mul(Function):
        def __init__(self, inputs):
            super().__init__(inputs=inputs, function=type(self).__name__)

        # Should not be called directly, prefer to use the __call__ method directly or indirectly
        def _forward(self):
            self.output.value = Tensor.Mul.mul(
                self.inputs[0].get_value(), self.inputs[1].get_value()
            )

        # Should not be called directly, prefer to use the __call__ method directly or indirectly
        # Should only be called with auto_grad=True
        def _backward(self):
            if Tensor.get_auto_grad() and self.inputs_requires_grad:
                # ∂L/∂output = 1
                self.output.grad = np.ones_like(
                    self.output.grad, dtype=Tensor.default_dtype
                )
                # ∂L/∂x = ∂L/∂output * y
                if self.inputs[0].requires_grad:
                    self.inputs[0].grad = Tensor.Add.add(
                        self.inputs[0].grad,
                        Tensor.Mul.mul(self.inputs[1].get_value(), self.output.grad),
                    )
                # ∂L/∂y = ∂L/∂output * x
                if self.inputs[1].requires_grad:
                    self.inputs[1].grad = Tensor.Add.add(
                        self.inputs[1].grad,
                        Tensor.Mul.mul(self.inputs[0].get_value(), self.output.grad),
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
            super().__init__(inputs=inputs, function=type(self).__name__)

        # Should not be called directly, prefer to use the __call__ method directly or indirectly
        def _forward(self):
            self.output.value = Tensor.Matmul.matmul(
                self.inputs[0].get_value(), self.inputs[1].get_value()
            )

        # Should not be called directly, prefer to use the __call__ method directly or indirectly
        # Should only be called with auto_grad=True
        def _backward(self):
            if Tensor.get_auto_grad() and self.inputs_requires_grad:
                # ∂L/∂output = 1
                self.output.grad = np.ones_like(
                    self.output.grad, dtype=Tensor.default_dtype
                )
                # ∂L/∂x = ∂L/∂output @ (y^T) (where y^T is the transpose of y)
                if self.inputs[0].requires_grad:
                    self.inputs[0].grad = self.inputs[0].grad + (
                        self.output.grad @ self.inputs[1].get_value().T
                    )
                # ∂L/∂y = (x^T) @ ∂L/∂output (where x^T is the transpose of x)
                if self.inputs[1].requires_grad:
                    self.inputs[1].grad = self.inputs[1].grad + (
                        self.inputs[0].get_value().T @ self.output.grad
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
            output = x @ y
            if not isinstance(output, np.ndarray):
                # Must be a scalar
                output = np.array([output])
            debug_print("output:", output)
            return output

    class Div(Function):
        def __init__(self, inputs):
            super().__init__(inputs=inputs, function=type(self).__name__)

        # Should not be called directly, prefer to use the __call__ method directly or indirectly
        def _forward(self):
            self.output.value = Tensor.Div.div(
                self.inputs[0].get_value(), self.inputs[1].get_value()
            )

        # Should not be called directly, prefer to use the __call__ method directly or indirectly
        # Should only be called with auto_grad=True
        def _backward(self):
            if Tensor.get_auto_grad() and self.inputs_requires_grad:
                # ∂L/∂output = 1
                self.output.grad = np.ones_like(
                    self.output.grad, dtype=Tensor.default_dtype
                )
                # ∂L/∂x = (1/y) * ∂L/∂output
                if self.inputs[0].requires_grad:
                    self.inputs[0].grad = self.inputs[0].grad + (
                        self.output.grad / self.inputs[1].get_value()
                    )
                # ∂L/∂y = (-x / (y^2)) * ∂L/∂output
                if self.inputs[1].requires_grad:
                    self.inputs[1].grad = (
                        self.inputs[1].grad
                        + (
                            (self.inputs[0].get_value() * -1)
                            / (self.inputs[1].get_value() ** 2)
                        )
                    ) * self.output.grad
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

    class IntDiv(Function):
        def __init__(self, inputs):
            super().__init__(inputs=inputs, function=type(self).__name__)

        # Should not be called directly, prefer to use the __call__ method directly or indirectly
        def _forward(self):
            self.output.value = Tensor.IntDiv.intdiv(
                self.inputs[0].get_value(), self.inputs[1].get_value()
            )

        # Should not be called directly, prefer to use the __call__ method directly or indirectly
        # Should only be called with auto_grad=True
        def _backward(self):
            raise Exception("IntDiv backward not implemented")

        @staticmethod
        def intdiv(
            x: Union["Tensor", np.ndarray], y: Union["Tensor", np.ndarray]
        ) -> np.ndarray:
            debug_print("intdiv x:", x)
            debug_print("intdiv y:", y)
            if isinstance(x, Tensor):
                x = x.get_value()
            if isinstance(y, Tensor):
                y = y.get_value()
            # Perform the division
            output = x // y
            if not isinstance(output, np.ndarray):
                # Must be a scalar
                output = np.array([output])
            debug_print("output:", output)
            return output

    class Neg(Function):
        def __init__(self, inputs):
            super().__init__(inputs=inputs, function=type(self).__name__)

        # Should not be called directly, prefer to use the __call__ method directly or indirectly
        def _forward(self):
            self.output.value = Tensor.Neg.neg(self.inputs[0].get_value())

        # Should not be called directly, prefer to use the __call__ method directly or indirectly
        # Should only be called with auto_grad=True
        def _backward(self):
            if Tensor.get_auto_grad() and self.inputs_requires_grad:
                # ∂L/∂output = 1
                self.output.grad = np.ones_like(
                    self.output.grad, dtype=Tensor.default_dtype
                )
                # ∂L/∂x = ∂L/∂output * -1
                if self.inputs[0].requires_grad:
                    self.inputs[0].grad = Tensor.Add.add(
                        self.inputs[0].grad, Tensor.Neg.neg(self.output.grad)
                    )
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

    def __add__(self, other: Union[TensorLikeForward, ScalarLikeForward]) -> "Tensor":
        return Tensor.Add([self, other])()

    def __radd__(self, other: Union[TensorLikeForward, ScalarLikeForward]) -> "Tensor":
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return other + self

    def __sub__(self, other: Union[TensorLikeForward, ScalarLikeForward]) -> "Tensor":
        return Tensor.Sub([self, other])()

    def __rsub__(self, other: Union[TensorLikeForward, ScalarLikeForward]) -> "Tensor":
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return other - self

    def __mul__(self, other: Union[TensorLikeForward, ScalarLikeForward]) -> "Tensor":
        return Tensor.Mul([self, other])()

    def __rmul__(self, other: Union[TensorLikeForward, ScalarLikeForward]) -> "Tensor":
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return other * self

    def __truediv__(
        self, other: Union[TensorLikeForward, ScalarLikeForward]
    ) -> "Tensor":
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor.Div([self, other])()

    def __rtruediv__(
        self, other: Union[TensorLikeForward, ScalarLikeForward]
    ) -> "Tensor":
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return other / self

    def __floordiv__(
        self, other: Union[TensorLikeForward, ScalarLikeForward]
    ) -> "Tensor":
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor.IntDiv([self, other])()

    def __rfloordiv__(
        self, other: Union[TensorLikeForward, ScalarLikeForward]
    ) -> "Tensor":
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return other // self

    def __matmul__(
        self, other: Union[TensorLikeForward, ScalarLikeForward]
    ) -> "Tensor":
        # Perform the dot product for vectors and matrix multiplication for matrices
        if Tensor.is_flat(self) and Tensor.is_flat(other):
            return Tensor.Dot([self, other])()
        else:
            return Tensor.Matmul([self, other])()

    def __rmatmul__(
        self, other: Union[TensorLikeForward, ScalarLikeForward]
    ) -> "Tensor":
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return other @ self

    def __neg__(self) -> "Tensor":
        return Tensor.Neg([self])()


class Scalar(Tensor):
    class InvalidScalarOperation(Exception):
        # Takes the function name and exception is "Cannot __func__ a scalar"
        def __init__(self, func_name: str):
            super().__init__(f"Cannot perform {func_name} on a scalar")

    def __init__(self, value: ScalarLikeForward, requires_grad: bool = False):
        # Assert that the value is a scalar
        assert isinstance(value, ScalarLikeForward), "Value must be a scalar"
        super().__init__(value=value, shape=(), requires_grad=requires_grad)

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
