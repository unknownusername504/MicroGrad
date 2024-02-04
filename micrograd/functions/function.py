from typing import List
import numpy as np
from micrograd.tensors.tensor import Tensor
from micrograd.utils.debug_utils import debug_print


class Function:
    def __init__(self, inputs: List[Tensor]):
        self.inputs = inputs
        self.output = None

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    @staticmethod
    def chunk(
        args: List[Tensor], num_threads: int, func_type: type
    ) -> List["Function"]:
        debug_print("chunk args:", args)
        debug_print("chunk num_threads:", num_threads)
        if len(args) != 2:
            raise Exception("Add function requires 2 arguments")
        # We want to split each tensor into chunks of the same size, number, and shape
        chunks = []
        # Determine the output shape based on the input shapes
        output_shape = Tensor.get_output_shape(args[0], args[1])
        # Detemine the output_type based on the input types
        output_tensor_type = Tensor.get_output_tensor_type(args[0], args[1])
        output_data_type = Tensor.get_output_data_type(args[0], args[1])
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
                    for k in range(this_chunk_size):
                        index = start + k
                        unravel_index = np.unravel_index(index, shape)
                        debug_print("unravel_index:", unravel_index)
                        chunk[k] = arr[unravel_index].get_value()
                    debug_print("chunk before:", chunk)
                    chunk = output_tensor_type(
                        shape=(this_chunk_size,),
                        value=chunk,
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
            # Create the op function
            func = func_type(inputs)
            # Append the new Add object
            returns.append(func)
        debug_print("returns from chunk:", returns)
        return returns

    @staticmethod
    def reduce(results: List[Tensor], output: Tensor):
        debug_print("reduce results:", results)
        debug_print("reduce output:", output)
        # We want to reduce the results into one result
        # We will assign the chunks to the output tensor
        # Get the number of chunks
        num_chunks = len(results)
        debug_print("num_chunks:", num_chunks)
        output_index = 0
        debug_print("output starting reduce:", output)
        output_shape = output.shape
        for i in range(num_chunks):
            debug_print("chunk num:", i)
            # Get the chunk output
            chunk_output = results[i]
            debug_print("chunk_output:", chunk_output)
            # Get the shape of the chunk_output
            shape = chunk_output.shape
            # Get the size of the chunk_output
            size = np.prod(shape)
            debug_print("chunk size:", size)
            if size == 0:
                continue
            elif len(output_shape) == 1:
                # Assign the chunk_output to the output tensor
                output[output_index : output_index + size] = chunk_output
            else:
                for index in range(output_index, output_index + size):
                    debug_print("index:", index)
                    output_tuple = np.unravel_index(index, output_shape)
                    debug_print("output_tuple:", output_tuple)
                    # Assign the chunk_output at index to the output tensor
                    output[output_tuple] = chunk_output[index - output_index]
            # Update the output_index
            output_index += size
        debug_print("output ending reduce:", output)

    def __call__(self):
        result = self.forward()
        if Tensor.auto_grad:
            self.backward()
        return result
