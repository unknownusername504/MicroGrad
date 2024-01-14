from typing import List, Tuple, Union
import numpy as np


class Tensor:
    def __init__(self, shape, dtype, value=None, requires_grad=False):
        self.shape = shape
        self.dtype = dtype
        # If the value is None then create a new numpy array of zeros
        if value is None:
            self.value = np.zeros(shape, dtype=dtype)
        else:
            self.value = value
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = None
        self.children = {}
        self.parents = {}

    # Length of the tensor
    def __len__(self):
        return len(self.value)

    # Casting the tensor dytpe
    def astype(self, dtype):
        self.value = self.value.astype(dtype)
        self.dtype = dtype
        return self

    # Slice the tensor
    def __getitem__(self, key: Union[int, slice]) -> "Tensor":
        # If the key is an integer
        if type(key) is int:
            # Make into a slice
            key = slice(key, key + 1)
        # Make sure the key range is valid
        if key.start < 0 or key.stop > len(self.value):
            raise Exception("Invalid key range")
        tensor_type = type(self)
        # Get the value
        value_slice = self.value[key]
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
        key: Union[int, slice],
        value: Union[List, np.ndarray, "Tensor", np.number, int, float],
    ):
        # If the key is an integer
        if type(key) is int:
            # Make into a slice
            key = slice(key, key + 1)
        # Make sure the key range is valid
        if key.start < 0 or key.stop > len(self.value):
            raise Exception("Invalid key range")
        # Flatten the incoming tensor value and ensure it is a numpy array of lesser or equal size
        if type(value) is not np.ndarray:
            value = np.array(value)
        value = value.flatten()
        # Ensure that the value is compatible with the tensor dtype with an attempt to cast
        try:
            value = value.astype(self.dtype)
        except:
            raise Exception("Incompatible value dtype")
        # Ensure the value is the same size as the key range
        if len(value) != (key.stop - key.start):
            raise Exception("Invalid value size")
        # Set the value
        self.value[key] = value

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
    def reshape(self, shape):
        self.value = self.value.reshape(shape)
        self.shape = shape
        return self

    # Transpose the tensor
    def transpose(self, axes=None):
        self.value = self.value.transpose(axes)
        self.shape = self.value.shape
        return self

    @staticmethod
    def can_broadcast(shape1, shape2):
        if shape1 == shape2:
            return True
        shape1, shape2 = list(shape1), list(shape2)
        # We want to see if the tensors can be broadcasted
        # Get the number of dimensions
        num_dims1 = len(shape1)
        num_dims2 = len(shape2)
        # Get the number of dimensions to pad
        num_dims_pad = abs(num_dims1 - num_dims2)
        # Get the number of dimensions to add
        num_dims_add = max(num_dims1, num_dims2) - num_dims_pad
        # Get the number of dimensions to broadcast
        num_dims_broadcast = min(num_dims1, num_dims2)
        # Loop through the dimensions to pad
        for _ in range(num_dims_pad):
            # Get the dimension
            dim1 = shape1.pop()
            dim2 = shape2.pop()
            # If the dimensions are not equal
            if dim1 != dim2:
                # Return False
                return False
        # Loop through the dimensions to add
        for _ in range(num_dims_add):
            # Get the dimension
            dim1 = shape1.pop()
            dim2 = shape2.pop()
            # If the dimensions are not equal
            if dim1 != dim2:
                # If one of the dimensions is 1
                if dim1 == 1 or dim2 == 1:
                    # Continue
                    continue
                else:
                    # Return False
                    return False
        # Loop through the dimensions to broadcast
        for _ in range(num_dims_broadcast):
            # Get the dimension
            dim1 = shape1.pop()
            dim2 = shape2.pop()
            # If the dimensions are not equal
            if dim1 != dim2:
                # If one of the dimensions is 1
                if dim1 == 1 or dim2 == 1:
                    # Continue
                    continue
                else:
                    # Return False
                    return False
        # Return True
        return True

    @staticmethod
    def get_output_shape(x: "Tensor", y: "Tensor") -> Tuple[int, ...]:
        if not Tensor.can_broadcast(x.shape, y.shape):
            raise Exception("Cannot broadcast tensors of different shapes")
        if x.shape == y.shape:
            return x.shape
        # We want to determine the output shape based on the input shapes
        # We need to see if the operation is element-wise or can be broadcasted
        # Get the shapes of the tensors
        shape1, shape2 = list(x.shape), list(y.shape)
        # Get the number of dimensions
        num_dims1 = len(shape1)
        num_dims2 = len(shape2)
        # Get the number of dimensions to pad
        num_dims_pad = abs(num_dims1 - num_dims2)
        # Get the number of dimensions to add
        num_dims_add = max(num_dims1, num_dims2) - num_dims_pad
        # Get the number of dimensions to broadcast
        num_dims_broadcast = min(num_dims1, num_dims2)
        # Get the output shape
        output_shape = []
        # Loop through the dimensions to pad
        for _ in range(num_dims_pad):
            # Append the dimension
            output_shape.append(1)
        # Loop through the dimensions to add
        for _ in range(num_dims_add):
            # Get the dimension
            dim1 = shape1.pop()
            dim2 = shape2.pop()
            # If the dimensions are not equal
            if dim1 != dim2:
                # If one of the dimensions is 1
                if dim1 == 1:
                    # Append the dimension
                    output_shape.append(dim2)
                elif dim2 == 1:
                    # Append the dimension
                    output_shape.append(dim1)
                else:
                    # Raise an exception
                    raise Exception("Cannot add tensors of different shapes")
            else:
                # Append the dimension
                output_shape.append(dim1)
        # Loop through the dimensions to broadcast
        for _ in range(num_dims_broadcast):
            # Get the dimension
            dim1 = shape1.pop()
            dim2 = shape2.pop()
            # If the dimensions are not equal
            if dim1 != dim2:
                # If one of the dimensions is 1
                if dim1 == 1:
                    # Append the dimension
                    output_shape.append(dim2)
                elif dim2 == 1:
                    # Append the dimension
                    output_shape.append(dim1)
                else:
                    # Raise an exception
                    raise Exception("Cannot add tensors of different shapes")
            else:
                # Append the dimension
                output_shape.append(dim1)
        # Return the output shape
        return tuple(output_shape)

    @staticmethod
    def broadcast(x: "Tensor", y: "Tensor") -> Tuple["Tensor", "Tensor"]:
        print("x:", x)
        print("y:", y)
        # We want to broadcast the tensors
        # Get the number of dimensions
        output_shape = Tensor.get_output_shape(x, y)
        print("output_shape:", output_shape)
        # Create the output tensors
        x_out = x.reshape(output_shape)
        y_out = y.reshape(output_shape)
        # Return the output tensors
        return x_out, y_out
