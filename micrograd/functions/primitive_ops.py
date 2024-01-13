from typing import List
import numpy as np

from micrograd.functions.wave_process import Function
from micrograd.tensors.tensor import Tensor

class Add(Function):
    def __init__(self, inputs, output):
        super().__init__(inputs, output)
        # Set the gradient function
        self.inputs[0].grad_fn = self
        self.inputs[1].grad_fn = self
        self.output.grad_fn = self
    
    def forward(self):
        self.output.value = self.add(self.inputs[0].value, self.inputs[1].value)
    
    def backward(self):
        self.inputs[0].grad = self.add(self.inputs[0].grad, self.output.grad)
        self.inputs[1].grad = self.add(self.inputs[1].grad, self.output.grad)
    
    def can_broadcast(self, shape1, shape2):
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
    
    def broadcast(self, x, y):
        # We want to broadcast the tensors
        # Get the number of dimensions
        num_dims1 = len(x.shape)
        num_dims2 = len(y.shape)
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
            # Get the dimension
            dim1 = x.shape.pop()
            dim2 = y.shape.pop()
            # If the dimensions are not equal
            if dim1 != dim2:
                # Raise an exception
                raise Exception('Cannot add tensors of different shapes')
        # Loop through the dimensions to add
        for _ in range(num_dims_add):
            # Get the dimension
            dim1 = x.shape.pop()
            dim2 = y.shape.pop()
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
                    raise Exception('Cannot add tensors of different shapes')
            else:
                # Append the dimension
                output_shape.append(dim1)
        # Loop through the dimensions to broadcast
        for _ in range(num_dims_broadcast):
            # Get the dimension
            dim1 = x.shape.pop()
            dim2 = y.shape.pop()
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
                    raise Exception('Cannot add tensors of different shapes')
            else:
                # Append the dimension
                output_shape.append(dim1)
        # Reverse the output shape
        output_shape.reverse()
        # Create the output tensors
        x_out = type(x)(output_shape, x)
        y_out = type(y)(output_shape, y)
        # Return the output tensors
        return x_out, y_out
    
    def add(self, x, y):
        # If the tensors are not the same shape
        if x.shape != y.shape:
            # If the tensors can be broadcasted
            if self.can_broadcast(x.shape, y.shape):
                # Broadcast the tensors
                x, y = self.broadcast(x, y)
            else:
                # Raise an exception
                raise Exception('Cannot add tensors of different shapes')
        # Schedule the job to the wave process worker
        # Return the output tensor
        return self.process(x, y)
    
    @staticmethod
    def get_output_shape(args):
        # We want to determine the output shape based on the input shapes
        # We need to see if the operation is element-wise or can be broadcasted
        # Get the shapes of the tensors
        shape1 = args[0].shape
        shape2 = args[1].shape
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
                    raise Exception('Cannot add tensors of different shapes')
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
                    raise Exception('Cannot add tensors of different shapes')
            else:
                # Append the dimension
                output_shape.append(dim1)
        # Reverse the output shape
        output_shape.reverse()
        return output_shape
    
    @staticmethod
    def chunk(args, num_threads) -> List[Function]:
        # We want to split each tensor into chunks of the same size, number, and shape
        chunks = [[], []]
        # Get the number of chunks
        num_chunks = num_threads
        for i in range(len(args)):
            # Get the shape of the tensors
            shape = args[i].shape
            # Get the size of the tensors
            size = np.prod(shape)
            # Get the chunk size
            chunk_size = size // num_chunks
            # Get the remainder
            remainder = size % num_chunks
            # Get the start index of the chunk
            start = 0
            # Get the end index of the chunk
            end = chunk_size
            # Loop through the chunks
            for _ in range(num_chunks):
                # Get the chunk
                chunk = args[i][start:end]
                # Append the chunk
                chunks[i].append(chunk)
                # Increment the start index
                start += chunk_size
                # Increment the end index
                end += chunk_size
                # If there is a remainder
                if remainder > 0:
                    # Increment the end index
                    end += 1
                    # Decrement the remainder
                    remainder -= 1
        # Determine the output shape based on the input shapes
        output_shape = Add.get_output_shape(args)
        returns = []
        # Create a new Add object for each chunk
        for i in range(num_chunks):
            # Create the output tensor
            output = type(chunks[0][0])(output_shape)
            inputs = [type(chunks[0][i])(chunks[0][i].shape, chunks[0][i]) for i in range(len(chunks[0]))]
            # Create the add function
            add = Add(inputs, output)
            # Append the new Add object
            returns.append(add)
        return returns

    @staticmethod
    def reduce(results: List[Function], output: Tensor):
        # We want to reduce the results into one result
        # We will assign the chunks to the output tensor
        # Get the number of chunks
        num_chunks = len(results)
        output_index = 0
        for i in range(num_chunks):
            # Get the chunk output
            chunk_output = results[i].output
            # Get the shape of the chunk_output
            shape = chunk_output.shape
            # Get the size of the chunk_output
            size = np.prod(shape)
            # Assign the chunk_output to the output tensor
            output.value[output_index:output_index + size] = chunk_output.value
            # Increment the output index
            output_index += size
        return [output]

class Sub(Function):
    def __init__(self, inputs, output):
        super().__init__(inputs, output)
    
    def forward(self):
        self.output.value = self.sub(self.inputs[0].value, self.inputs[1].value)
    
    def backward(self):
        self.inputs[0].grad = self.add(self.inputs[0].grad, self.output.grad)
        self.inputs[1].grad = self.sub(self.inputs[1].grad, self.output.grad)

class Dot(Function):
    def __init__(self, inputs, output):
        super().__init__(inputs, output)
    
    def forward(self):
        self.output.value = self.dot(self.inputs[0].value, self.inputs[1].value)
    
    def backward(self):
        self.inputs[0].grad = self.add(self.inputs[0].grad, self.dot(self.inputs[1].value, self.output.grad))
        self.inputs[1].grad = self.add(self.inputs[1].grad, self.dot(self.inputs[0].value, self.output.grad))

class Matmul(Function):
    def __init__(self, inputs, output):
        super().__init__(inputs, output)
    
    def forward(self):
        self.output.value = self.matmul(self.inputs[0].value, self.inputs[1].value)
    
    def backward(self):
        self.inputs[0].grad = self.add(self.inputs[0].grad, self.matmul(self.output.grad, self.inputs[1].value.T))
        self.inputs[1].grad = self.add(self.inputs[1].grad, self.matmul(self.inputs[0].value.T, self.output.grad))