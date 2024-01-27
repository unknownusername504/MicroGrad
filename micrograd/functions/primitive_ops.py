from typing import List
import numpy as np

from micrograd.functions.wave_process import Function
from micrograd.tensors.tensor import Tensor
from micrograd.utils.debug_utils import debug_print


class Add(Function):
    def __init__(self, inputs, output):
        super().__init__(inputs, output)
        # Set the gradient function
        self.inputs[0].grad_fn = self
        self.inputs[1].grad_fn = self
        self.output.grad_fn = self

    def forward(self):
        self.output.value = self.add(self.inputs[0], self.inputs[1])
        return self.output.value

    def backward(self):
        self.inputs[0].grad = self.add(self.inputs[0], self.output)
        self.inputs[1].grad = self.add(self.inputs[1], self.output)

    def add(self, x, y):
        # If the tensors are not the same shape
        # Broadcast the tensors
        output_shape = Tensor.get_output_shape(x, y)
        x, y = Tensor.broadcast(x, y)
        debug_print("add x:", x)
        debug_print("add y:", y)
        # Schedule the job to the wave process worker
        # Return the output tensor
        output = self.process(x, y)
        output = output.reshape(output_shape)
        debug_print("output:", output)
        return output

    @staticmethod
    def chunk(args: List[Tensor], num_threads: int) -> List[Function]:
        debug_print("chunk args:", args)
        debug_print("chunk num_threads:", num_threads)
        if len(args) != 2:
            raise Exception("Add function requires 2 arguments")
        requires_grad = args[0].requires_grad
        # We want to split each tensor into chunks of the same size, number, and shape
        chunks = []
        # Determine the output shape based on the input shapes
        output_shape = Tensor.get_output_shape(args[0], args[1])
        # Detemine the output_type based on the input types
        output_tensor_type, output_data_type = Tensor.get_output_type(args[0], args[1])
        debug_print("output_shape:", output_shape)
        debug_print("output_tensor_type:", output_tensor_type)
        debug_print("output_data_type:", output_data_type)
        # Get the number of chunks
        num_chunks = min(num_threads, np.prod(output_shape))
        debug_print("num_chunks:", num_chunks)
        for i in range(len(args)):
            debug_print("i:", i)
            chunks.append([])
            # Get the shape of the tensors
            shape = args[i].shape
            # Get the size of the tensors
            size = np.prod(shape)
            # Get the chunk size
            chunk_size = size // num_chunks
            # Get the remainder
            remainder = size % num_chunks
            # Loop through the chunks
            for j in range(num_chunks):
                this_chunk_size = chunk_size + (1 if remainder > j else 0)
                # Get the start index
                start = j * chunk_size + min(j, remainder)
                # Get the end index
                end = start + this_chunk_size
                debug_print("start of chunk:", start)
                debug_print("end of chunk:", end)
                chunk = np.zeros((this_chunk_size,), dtype=output_data_type)
                # Decide how to get the values based on the shape of the tensor
                if len(shape) == 1:
                    # Get the chunk
                    chunk_value = args[i][start:end].get_value()
                    chunk[start:end] = chunk_value
                else:
                    # Get the chunk
                    arr = args[i]
                    debug_print("arr:", arr)
                    num_dims = len(shape)
                    for k in range(this_chunk_size):
                        index = start + k
                        remainder = index
                        indices = []
                        for l in range(num_dims):
                            dim = shape[l]
                            indices.append(remainder % dim)
                            remainder //= dim
                        indices = tuple(indices)
                        debug_print("indices:", indices)
                        chunk_value = arr[indices].get_value()
                        chunk[k] = chunk_value
                    debug_print("chunk before:", chunk)
                    chunk = output_tensor_type(
                        shape=(this_chunk_size,),
                        value=chunk,
                        requires_grad=requires_grad,
                    )
                debug_print("chunk:", chunk)
                # Append the chunk
                chunks[i].append(chunk)
        debug_print("chunks:", chunks)
        returns = []
        # Create a new Add object for each chunk
        for i in range(num_chunks):
            debug_print("Trying to create input tensors")
            inputs = [chunks[0][i], chunks[1][i]]
            debug_print("inputs:", inputs)
            # Create the output tensor
            debug_print("Trying to create output tensor")
            this_output_shape = Tensor.get_output_shape(inputs[0], inputs[1])
            debug_print("this_output_shape:", this_output_shape)
            output = np.zeros(shape=this_output_shape, dtype=output_data_type)
            output = output_tensor_type(
                shape=this_output_shape, value=output, requires_grad=requires_grad
            )
            debug_print("output:", output)
            # Create the add function
            add = Add(inputs, output)
            # Append the new Add object
            returns.append(add)
        debug_print("returns from chunk:", returns)
        return returns

    @staticmethod
    def reduce(results: List[Function], output: Tensor) -> List[Tensor]:
        debug_print("reduce results:", results)
        # We want to reduce the results into one result
        # We will assign the chunks to the output tensor
        # Get the number of chunks
        num_chunks = len(results)
        output_index = 0
        debug_print("output:", output)
        for i in range(num_chunks):
            # Get the chunk output
            chunk_output = results[i].output
            # Get the shape of the chunk_output
            shape = chunk_output.shape
            # Get the size of the chunk_output
            size = np.prod(shape)
            indices = [index for index in range(output_index, output_index + size)]
            output_tuple = np.unravel_index(indices, shape)
            # Assign the chunk_output to the output tensor
            output[output_tuple] = chunk_output.value
            # Increment the output index
            output_index += size
        debug_print("output:", output)
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
