import numpy as np
from micrograd.functions.primitive_ops import Add, Sub, Dot, Matmul
from micrograd.tensors.tensor import Tensor


# Class that implements a tensor of unsigned 8-bit integers
# All operations are quantized and performed on the CPU
class TensorU8(Tensor):
    def __init__(self, shape, value=None, requires_grad=False):
        self.dtype = np.uint8
        super().__init__(
            shape=shape, dtype=self.dtype, value=value, requires_grad=requires_grad
        )

    def __add__(self, other):
        if isinstance(other, TensorU8):
            # Create the output tensor
            output = TensorU8(self.shape)
            # Create the add function
            add = Add([self, other], output)
            # Set the gradient function
            self.grad_fn = add
            other.grad_fn = add
            # Call the forward function
            add.forward()
            # Return the output tensor
            return output
        else:
            return self.add(self, other)

    def __radd__(self, other):
        return other + self

    def __sub__(self, other):
        if isinstance(other, TensorU8):
            # Create the output tensor
            output = TensorU8(self.shape)
            # Create the sub function
            sub = Sub([self, other], output)
            # Set the gradient function
            self.grad_fn = sub
            other.grad_fn = sub
            # Call the forward function
            sub.forward()
            # Return the output tensor
            return output
        else:
            return self.sub(self, other)

    def __rsub__(self, other):
        return other - self

    def __mul__(self, other):
        if isinstance(other, TensorU8):
            # Create the output tensor
            output = TensorU8(self.shape)
            # Create the dot function
            dot = Dot([self, other], output)
            # Set the gradient function
            self.grad_fn = dot
            other.grad_fn = dot
            # Call the forward function
            dot.forward()
            # Return the output tensor
            return output
        else:
            return self.mul(self, other)

    def __rmul__(self, other):
        return other * self

    def __matmul__(self, other):
        if isinstance(other, TensorU8):
            # Create the output tensor
            output = TensorU8(self.shape)
            # Create the matmul function
            matmul = Matmul([self, other], output)
            # Set the gradient function
            self.grad_fn = matmul
            other.grad_fn = matmul
            # Call the forward function
            matmul.forward()
            # Return the output tensor
            return output
        else:
            return self.matmul(self, other)

    def __rmatmul__(self, other):
        return other @ self

    def add(self, x, y):
        return np.add(x, y, dtype=np.uint8)

    def sub(self, x, y):
        return np.subtract(x, y, dtype=np.uint8)

    def mul(self, x, y):
        return np.multiply(x, y, dtype=np.uint8)

    def dot(self, x, y):
        return np.dot(x, y, dtype=np.uint8)

    def matmul(self, x, y):
        return np.matmul(x, y, dtype=np.uint8)
