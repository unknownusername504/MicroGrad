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
        self.output.value = self.add(self.inputs[0], self.inputs[1])

    def backward(self):
        self.inputs[0].grad = self.add(self.inputs[0], self.output)
        self.inputs[1].grad = self.add(self.inputs[1], self.output)

    def add(self, x, y):
        # If the tensors are not the same shape
        # Broadcast the tensors
        output_shape = Tensor.get_output_shape(x, y)
        x, y = Tensor.broadcast(x, y)
        # Schedule the job to the wave process worker
        # Return the output tensor
        output = self.process(x, y)
        output = output.reshape(output_shape)
        print("output:", output)
        return output

    @staticmethod
    def chunk(args: List[Tensor], num_threads: int) -> List[Function]:
        if len(args) != 2:
            raise Exception("Add function requires 2 arguments")
        tensor_type = type(args[0])
        requires_grad = args[0].requires_grad
        # We want to split each tensor into chunks of the same size, number, and shape
        chunks = [[], []]
        # Determine the output shape based on the input shapes
        output_shape = Tensor.get_output_shape(args[0], args[1])
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
        returns = []
        # Create a new Add object for each chunk
        for i in range(num_chunks):
            # Create the output tensor
            print("Trying to create output tensor with shape:", output_shape)
            output = tensor_type(shape=output_shape, requires_grad=requires_grad)
            print("output:", output)
            print("Trying to create input tensors")
            inputs = [chunks[0][i], chunks[1][i]]
            print("inputs:", inputs)
            # Create the add function
            add = Add(inputs, output)
            # Append the new Add object
            returns.append(add)
        print("returns:", returns)
        return returns

    @staticmethod
    def reduce(results: List[Function], output: Tensor):
        print("\nreduce results:", results)
        # We want to reduce the results into one result
        # We will assign the chunks to the output tensor
        # Get the number of chunks
        num_chunks = len(results)
        output_index = 0
        print("output:", output)
        for i in range(num_chunks):
            # Get the chunk output
            chunk_output = results[i].output
            # Get the shape of the chunk_output
            shape = chunk_output.shape
            # Get the size of the chunk_output
            size = np.prod(shape)
            # Assign the chunk_output to the output tensor
            output[output_index : output_index + size] = chunk_output.value
            # Increment the output index
            output_index += size
        print("output:", output)
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
        self.inputs[0].grad = self.add(
            self.inputs[0].grad, self.dot(self.inputs[1].value, self.output.grad)
        )
        self.inputs[1].grad = self.add(
            self.inputs[1].grad, self.dot(self.inputs[0].value, self.output.grad)
        )


class Matmul(Function):
    def __init__(self, inputs, output):
        super().__init__(inputs, output)

    def forward(self):
        self.output.value = self.matmul(self.inputs[0].value, self.inputs[1].value)

    def backward(self):
        self.inputs[0].grad = self.add(
            self.inputs[0].grad, self.matmul(self.output.grad, self.inputs[1].value.T)
        )
        self.inputs[1].grad = self.add(
            self.inputs[1].grad, self.matmul(self.inputs[0].value.T, self.output.grad)
        )
