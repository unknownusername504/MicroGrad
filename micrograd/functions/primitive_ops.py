from typing import Union
import numpy as np

from micrograd.functions.function import Function
from micrograd.tensors.tensor import Tensor
from micrograd.utils.debug_utils import debug_print


class Add(Function):
    def __init__(self, inputs):
        super().__init__(inputs)
        # Set the gradient function
        self.inputs[0].grad_fn = self
        self.inputs[1].grad_fn = self
        x = self.inputs[0]
        y = self.inputs[1]
        output_tensor_type = Tensor.get_output_tensor_type(x, y)
        output_shape = Tensor.get_output_shape(x, y)
        self.output = output_tensor_type(shape=output_shape)
        self.output.grad_fn = self

    # Should not be called directly, prefer to use the __call__ method directly or indirectly
    def forward(self):
        self.output.value = self.add(
            self.inputs[0].get_value(), self.inputs[1].get_value()
        )
        return self.output

    # Should not be called directly, prefer to use the __call__ method directly or indirectly
    # Should only be called with auto_grad=True
    def backward(self):
        if Tensor.auto_grad:
            self.inputs[0].grad = self.add(self.inputs[0].grad, self.output.grad)
            self.inputs[1].grad = self.add(self.inputs[1].grad, self.output.grad)
        else:
            raise Exception("Backward should only be called with auto_grad=True")

    def add(
        self, x: Union[Tensor, np.ndarray], y: Union[Tensor, np.ndarray]
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
            output = np.array(output)
        debug_print("output:", output)
        return output


class Sub(Function):
    def __init__(self, inputs):
        super().__init__(inputs)
        # Set the gradient function
        self.inputs[0].grad_fn = self
        self.inputs[1].grad_fn = self
        x = self.inputs[0]
        y = self.inputs[1]
        output_tensor_type = Tensor.get_output_tensor_type(x, y)
        output_shape = Tensor.get_output_shape(x, y)
        self.output = output_tensor_type(shape=output_shape)
        self.output.grad_fn = self

    # Should not be called directly, prefer to use the __call__ method directly or indirectly
    def forward(self):
        self.output.value = self.sub(
            self.inputs[0].get_value(), self.inputs[1].get_value()
        )
        return self.output

    # Should not be called directly, prefer to use the __call__ method directly or indirectly
    # Should only be called with auto_grad=True
    def backward(self):
        if Tensor.auto_grad:
            self.inputs[0].grad = self.add(self.inputs[0].grad, self.output.grad)
            self.inputs[1].grad = self.sub(self.inputs[1].grad, self.output.grad)
        else:
            raise Exception("Backward should only be called with auto_grad=True")

    def sub(
        self, x: Union[Tensor, np.ndarray], y: Union[Tensor, np.ndarray]
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
            output = np.array(output)
        debug_print("output:", output)
        return output


class Dot(Function):
    def __init__(self, inputs):
        super().__init__(inputs)
        # Set the gradient function
        self.inputs[0].grad_fn = self
        self.inputs[1].grad_fn = self
        x = self.inputs[0]
        y = self.inputs[1]
        output_tensor_type = Tensor.get_output_tensor_type(x, y)
        output_shape = Tensor.get_output_shape(x, y)
        self.output = output_tensor_type(shape=output_shape)
        self.output.grad_fn = self

    # Should not be called directly, prefer to use the __call__ method directly or indirectly
    def forward(self):
        self.output.value = self.dot(
            self.inputs[0].get_value(), self.inputs[1].get_value()
        )
        return self.output

    # Should not be called directly, prefer to use the __call__ method directly or indirectly
    # Should only be called with auto_grad=True
    def backward(self):
        if Tensor.auto_grad:
            self.inputs[0].grad = self.add(
                self.inputs[0].grad,
                self.dot(self.inputs[1].get_value(), self.output.grad),
            )
            self.inputs[1].grad = self.add(
                self.inputs[1].grad,
                self.dot(self.inputs[0].get_value(), self.output.grad),
            )
        else:
            raise Exception("Backward should only be called with auto_grad=True")

    def dot(
        self, x: Union[Tensor, np.ndarray], y: Union[Tensor, np.ndarray]
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
            output = np.array(output)
        debug_print("output:", output)
        return output


class Matmul(Function):
    def __init__(self, inputs):
        super().__init__(inputs)
        # Set the gradient function
        self.inputs[0].grad_fn = self
        self.inputs[1].grad_fn = self
        x = self.inputs[0]
        y = self.inputs[1]
        output_tensor_type = Tensor.get_output_tensor_type(x, y)
        output_shape = Tensor.get_output_shape(x, y)
        self.output = output_tensor_type(shape=output_shape)
        self.output.grad_fn = self

    # Should not be called directly, prefer to use the __call__ method directly or indirectly
    def forward(self):
        self.output.value = self.matmul(
            self.inputs[0].get_value(), self.inputs[1].get_value()
        )
        return self.output

    # Should not be called directly, prefer to use the __call__ method directly or indirectly
    # Should only be called with auto_grad=True
    def backward(self):
        if Tensor.auto_grad:
            self.inputs[0].grad = self.add(
                self.inputs[0].grad,
                self.matmul(self.output.grad, self.inputs[1].get_value().T),
            )
            self.inputs[1].grad = self.add(
                self.inputs[1].grad,
                self.matmul(self.inputs[0].get_value().T, self.output.grad),
            )
        else:
            raise Exception("Backward should only be called with auto_grad=True")

    def matmul(
        self, x: Union[Tensor, np.ndarray], y: Union[Tensor, np.ndarray]
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
            output = np.array(output)
        debug_print("output:", output)
        return output
