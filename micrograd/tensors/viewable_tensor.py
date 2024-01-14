# Class for viewing non contiguous tensor slices without copying

from typing import List, Optional, Union
import unittest

import numpy as np
from micrograd.tensors.tensor import Tensor


class ViewableTensor(Tensor):
    def __init__(self, tensor: Tensor, slices: Optional[List[slice]] = None):
        self.viewable_shape = tensor.shape
        # Get the flattened tensor
        tensor.flatten()
        # Initialize the super class with the tensor's attributes
        super().__init__(
            shape=tensor.shape,
            dtype=tensor.dtype,
            value=tensor.value,
            requires_grad=tensor.requires_grad,
        )

        # A list of slices, which starts as the full tensor if not specified
        if slices is None:
            self.slices = [slice(0, len(self), 1)]
        else:
            # Check that the slices are valid
            for i, s in enumerate(slices):
                # Populate none values in the slices and sanitize the slices
                slices[i] = self.sanitize_slice(s)
            self.slices = slices
            self.combine_slices()

    # Populate none values in the slice for uniformity
    # Also ensure the stop is exactly on a step from start
    # Called in the constructor and when slicing views
    def sanitize_slice(self, this_slice: slice) -> slice:
        try:
            assert isinstance(this_slice, slice)
            # Assert that the range of the slice is valid
            assert (this_slice.start >= 0) and (this_slice.start <= len(self))
            assert (this_slice.stop <= len(self)) and (this_slice.stop >= 0)
            # Assert that abs(step) is not greater than the length of the slice
            assert abs(this_slice.step) <= (this_slice.stop - this_slice.start) and (
                this_slice.step != 0
            )
            # Assert that the slice is not empty
            assert (this_slice.start <= this_slice.stop) or (this_slice.step < 0)
        except AssertionError:
            raise ValueError("Invalid slice")
        start = 0 if this_slice.start is None else this_slice.start
        stop = len(self) if this_slice.stop is None else this_slice.stop
        step = 1 if this_slice.step is None else this_slice.step
        # If the mod of stop - start and step is not 0 then set the stop to the next step
        mod_stop_start = (stop - start) % step
        if mod_stop_start != 0:
            remainder = step - mod_stop_start
            stop += remainder
        # If the stop is equal to the start then set the step to 1 and the stop to the next step
        if stop == start:
            step = 1
            stop = start + step
        # If the abs of the step is not 1 and the slice would only have 1 element then set the step to 1
        if abs(step) != 1 and (stop - step) == start:
            step = 1
        return slice(start, stop, step)

    # Set the slices of the viewable tensor
    def set_slices(self, slices: List[slice]):
        for i, slice in enumerate(slices):
            # Populate none values in the slices and sanitize the slices
            slices[i] = self.sanitize_slice(slice)
        self.slices = slices
        self.combine_slices()

    # Redefine the len method to be symbolic
    def __len__(self):
        return np.prod(self.viewable_shape)

    # Redefine the getitem method to return a contiguous tensor
    def __getitem__(self, key: Union[int, slice]) -> "Tensor":
        # If the key is an integer
        if type(key) is int:
            # Make into a slice
            key = slice(key, key + 1)
        # Make sure the key range is valid
        if key.start < 0 or key.stop > len(self):
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

    def get_contiguous(self) -> "Tensor":
        # Return the contiguous tensor
        return self[0 : len(self)]

    # Redefine the reshape method to be symbolic
    def reshape(self, shape):
        # Check that the shape is valid
        if np.prod(shape) != len(self):
            raise Exception("Invalid shape")
        # Set the viewable shape
        self.viewable_shape = shape
        # Return the tensor
        return self

    # Redefine the flatten method to be symbolic
    def flatten(self):
        # Set the viewable shape
        self.viewable_shape = (len(self),)
        # Return the tensor
        return self

    # Attempt to combine slices to reduce the number of slices, does not attempt to go over 2 steps
    # Can work with negative steps
    # Is not guaranteed to find the optimal solution
    def combine_slices(self):
        # Populate the first slice
        new_slices = [
            slice(self.slices[0].start, self.slices[0].stop, self.slices[0].step)
        ]
        # Iterate through the slices
        for index in range(1, len(self.slices)):
            last_slice = new_slices[-1]
            next_slice = self.slices[index]
            # Slices can be combined if they have the same step and the stop of the first is the start of the second
            # Or if the length of the next slice is 1
            if (last_slice.step == next_slice.step) and (
                last_slice.stop == next_slice.start
            ):
                # Combine the slices
                new_slices[-1] = slice(
                    last_slice.start, next_slice.stop, last_slice.step
                )
            elif (abs(next_slice.stop - next_slice.start) // abs(next_slice.step)) == 1:
                # Combine the slices
                new_slices[-1] = slice(
                    last_slice.start,
                    next_slice.start + last_slice.step,
                    last_slice.step,
                )
            else:
                # Add the slice
                new_slices.append(next_slice)
        # Set the slices
        self.slices = new_slices

    # Redefine the transpose method to work with viewable tensors
    def transpose(self, axes=None):
        # If the axes are not specified then set it to the reverse of first axis
        if axes is None:
            axes = tuple(reversed(range(len(self.viewable_shape))))
        # Perform the transpose by modifying the slices
        # This needs to take into account the original shape of the viewable tensor and
        # the new shape of the viewable tensor to get the correct slices
        # Slices can also be any slice of the underlying tensor in any order and with overlaps or gaps
        # Slices can also have steps
        # Create a matrix of the indices to be used for the transpose
        matrix_post = np.indices(self.shape)
        # Populate the matrix_pre with the slices
        matrix_index = 0
        for slice in self.slices:
            # Get the start, stop, and step
            start = slice.start
            stop = slice.stop
            step = slice.step
            for slice_index in range((stop - start) // step):
                slice_index_real = (slice_index * step) + start
                # Set the slice
                matrix_post[matrix_index] = slice_index_real
                # Increment the matrix_index
                matrix_index += 1
        # Reshape the matrix_post to be the shape of the viewable tensor
        matrix_post = matrix_post.reshape(self.viewable_shape)
        # Transpose the post matrix
        matrix_post = matrix_post.transpose(axes)
        # Flatten the matrix_post
        matrix_post = matrix_post.flatten()
        # Iterate through the matrixes and update the slices
        # This is done by iterating through the matrix and creating contiguous slices
        self.slices = [
            slice(matrix_post[i], matrix_post[i + 1], 1)
            for i in range(len(matrix_post) - 1)
        ]
        # See if the slices can be combined
        self.combine_slices()
        # Set the viewable shape by transposing the axes
        self.viewable_shape = self.viewable_shape[axes]


# Unit tests for ViewableTensor
class TestViewableTensor(unittest.TestCase):
    def test_viewable_tensor(self):
        try:
            # Create the tensor
            tensor = Tensor(
                shape=(2, 2), dtype=np.uint8, value=np.array([[1, 2], [3, 4]])
            )
            # Create the viewable tensor
            viewable_tensor = ViewableTensor(tensor)
        except Exception as e:
            self.fail("Exception raised while creating ViewableTensor: " + str(e))

        try:
            # Test the len method
            self.assertEqual(len(viewable_tensor), 4)
            # Test the shape
            self.assertEqual(viewable_tensor.shape, (4,))
            # Test the dtype
            self.assertEqual(viewable_tensor.dtype, np.uint8)
            # Test the value
            self.assertEqual(viewable_tensor.value.tolist(), [1, 2, 3, 4])
            # Test the viewable shape
            self.assertEqual(viewable_tensor.viewable_shape, (2, 2))
            # Test the slices
            self.assertEqual(viewable_tensor.slices, [slice(0, 4, 1)])
        except Exception as e:
            self.fail("Exception raised while testing member variables: " + str(e))

        try:
            # Test the flatten method
            viewable_tensor.flatten()
            self.assertEqual(viewable_tensor.viewable_shape, (4,))
            self.assertEqual(viewable_tensor.slices, [slice(0, 4, 1)])
        except Exception as e:
            self.fail("Exception raised while testing flatten method: " + str(e))

        try:
            # Test the reshape method
            viewable_tensor.reshape((2, 2))
            self.assertEqual(viewable_tensor.viewable_shape, (2, 2))
            self.assertEqual(viewable_tensor.slices, [slice(0, 4, 1)])
        except Exception as e:
            self.fail("Exception raised while testing reshape method: " + str(e))

        try:
            # Test the transpose method
            viewable_tensor.transpose()
            self.assertEqual(viewable_tensor.viewable_shape, (2, 2))
            self.assertEqual(viewable_tensor.value.tolist(), [1, 2, 3, 4])
            # Transposed array will be [[1, 3], [2, 4]]
            self.assertEqual(
                viewable_tensor.get_contiguous().value.tolist(), [1, 3, 2, 4]
            )
            # Transpose back
            viewable_tensor.transpose()
            self.assertEqual(viewable_tensor.viewable_shape, (2, 2))
            self.assertEqual(viewable_tensor.value.tolist(), [1, 2, 3, 4])
            # Transposed array will be [[1, 2], [3, 4]]
            self.assertEqual(
                viewable_tensor.get_contiguous().value.tolist(), [1, 2, 3, 4]
            )
        except Exception as e:
            self.fail("Exception raised while testing transpose method: " + str(e))

        try:
            # Test the getitem method
            self.assertEqual(viewable_tensor[0].value.tolist(), [1])
            self.assertEqual(viewable_tensor[1].value.tolist(), [2])
            self.assertEqual(viewable_tensor[2].value.tolist(), [3])
            self.assertEqual(viewable_tensor[3].value.tolist(), [4])
            self.assertEqual(viewable_tensor[0:2].value.tolist(), [1, 2])
            self.assertEqual(viewable_tensor[2:4].value.tolist(), [3, 4])
            self.assertEqual(viewable_tensor[0:4:2].value.tolist(), [1, 3])
            self.assertEqual(viewable_tensor[1:4:2].value.tolist(), [2, 4])
            self.assertEqual(viewable_tensor[0:4:3].value.tolist(), [1, 4])
            self.assertEqual(viewable_tensor[0:4:4].value.tolist(), [1])
            self.assertEqual(viewable_tensor[0:4:5].value.tolist(), [1])
            self.assertEqual(viewable_tensor[0:5:1].value.tolist(), [1, 2, 3, 4])
        except Exception as e:
            self.fail("Exception raised while testing getitem method: " + str(e))

        try:
            # Test the sanitize_slice method
            self.assertEqual(
                viewable_tensor.sanitize_slice(slice(0, 4, 1)), slice(0, 4, 1)
            )
            self.assertEqual(
                viewable_tensor.sanitize_slice(slice(0, 0, 1)), slice(0, 1, 1)
            )
            self.assertEqual(
                viewable_tensor.sanitize_slice(slice(0, 1, 1)), slice(0, 1, 1)
            )
            self.assertEqual(
                viewable_tensor.sanitize_slice(slice(0, 1, -1)), slice(0, 1, 1)
            )
            self.assertEqual(
                viewable_tensor.sanitize_slice(slice(0, 2, -1)), slice(0, 2, -1)
            )
            self.assertEqual(
                viewable_tensor.sanitize_slice(slice(0, 1, 2)), slice(0, 1, 1)
            )
            self.assertEqual(
                viewable_tensor.sanitize_slice(slice(0, 3, 2)), slice(0, 4, 2)
            )
        except Exception as e:
            self.fail("Exception raised while testing sanitize_slice method: " + str(e))

        try:
            # Test the combine_slices method
            viewable_tensor.set_slices(
                [
                    slice(0, 1, 1),
                    slice(1, 2, 1),
                    slice(2, 3, 1),
                    slice(3, 4, 1),
                ]
            )
            viewable_tensor.combine_slices()
            self.assertEqual(viewable_tensor.slices, [slice(0, 4, 1)])

            viewable_tensor.set_slices(
                [
                    slice(0, 2, 1),
                    slice(2, 4, 1),
                ]
            )
            viewable_tensor.combine_slices()
            self.assertEqual(viewable_tensor.slices, [slice(0, 4, 1)])

            viewable_tensor.set_slices(
                [
                    slice(2, 4, 1),
                    slice(0, 2, 1),
                ]
            )
            viewable_tensor.combine_slices()
            self.assertEqual(viewable_tensor.slices, [slice(2, 4, 1), slice(0, 2, 1)])
            viewable_tensor.set_slices(
                [
                    slice(0, 4, 1),
                ]
            )
            self.assertEqual(viewable_tensor.slices, [slice(0, 4, 1)])
            viewable_tensor.combine_slices()
            self.assertEqual(viewable_tensor.slices, [slice(0, 4, 1)])
        except Exception as e:
            self.fail("Exception raised while testing combine_slices method: " + str(e))


def unittest_viewable_tensor():
    print("Running unit tests for ViewableTensor...")
    # Run the unit tests
    unittest.main()
    print("Finished unit tests for ViewableTensor")


if __name__ == "__main__":
    unittest_viewable_tensor()
