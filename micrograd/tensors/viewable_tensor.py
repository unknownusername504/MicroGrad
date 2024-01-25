# Class for viewing non contiguous tensor slices without copying

import copy
from typing import List, Optional, Union
import unittest

import numpy as np
from micrograd.tensors.tensor import Tensor


class ViewableTensor(Tensor):
    def __init__(self, tensor: Tensor, slices: Optional[List[slice]] = None):
        self.tensor_type = type(tensor)
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
            for i, this_slice in enumerate(slices):
                assert isinstance(this_slice, slice)
                # Populate none values in the slices and sanitize the slices
                slices[i] = self.simplify_slice(this_slice)
            self.slices = slices
            self.combine_slices()

        self.allow_shallow_copy = True
        self.allow_deep_copy = True
        self.allow_modify = True

        # Track shallow copies as a list
        self.shallow_copies = []

    def set_allow_shallow_copy(self, allow_shallow_copy: bool):
        self.allow_shallow_copy = allow_shallow_copy

    def set_allow_deep_copy(self, allow_deep_copy: bool):
        self.allow_deep_copy = allow_deep_copy

    def set_allow_modify(self, allow_modify: bool):
        self.allow_modify = allow_modify

    # Redefine copy to test if the copy is allowed
    def __copy__(self):
        if not self.allow_shallow_copy:
            raise Exception("Copy not allowed")
        # Create the copy with all the same attributes
        my_cls = type(self)
        shallow_copy = my_cls.__new__(my_cls)
        # Add this shallow copy to the list of shallow copies
        self.shallow_copies.append(shallow_copy)
        # Copy the attributes
        shallow_copy_dict = self.__dict__.copy()
        # Update the shallow copy
        shallow_copy.__dict__.update(shallow_copy_dict)
        return shallow_copy

    def __deepcopy__(self, memo: Optional[dict] = None):
        if not self.allow_deep_copy:
            raise Exception("Copy not allowed")
        deep_copy = type(self)(copy.deepcopy(self, memo))
        return deep_copy

    def copy(self):
        return self.__deepcopy__(self)

    # Populate none values in the slice for uniformity
    # Also ensure the stop is exactly on a step from start
    # Called in the constructor and when slicing views
    def simplify_slice(self, this_slice: slice) -> slice:
        try:
            start = 0 if this_slice.start is None else this_slice.start
            stop = len(self) if this_slice.stop is None else this_slice.stop
            step = 1 if this_slice.step is None else this_slice.step
            # Handle negative start/stop
            if start < 0:
                start = max((len(self) + start), 0)
            if stop < 0:
                stop = len(self) + stop
                if stop < 0:
                    if step > 0:
                        return slice(0, 0, 1)
                    else:
                        if start == 0:
                            return slice(0, 0, 1)
                        # Very negative stops behave like no stop but
                        # we emulate this with a just out of range stop
                        stop = -len(self) - 1
            # Handle greater than length start/stop
            if start > len(self):
                if step > 0:
                    start = len(self)
                else:
                    start = len(self) - 1
            if stop > len(self):
                stop = len(self)
            # Handle negative step
            if step < 0:
                # If the magnitude of the step is greater than the length of the slice then set the step to 1
                if abs(step) > abs(stop - start):
                    if start > stop:
                        return slice(start, start + 1, 1)
                    else:
                        return slice(0, 0, 1)
            # Check that the first step taken is inside the array
            if (
                (start == stop)
                or ((start < stop) and (step < 0))
                or ((start > stop) and (step > 0))
            ):
                return slice(0, 0, 1)
            num_steps = abs(stop - start) // abs(step)
            if (start + (num_steps * step)) < stop:
                num_steps += 1
            if num_steps == 0:
                return slice(0, 0, 1)
            # If the slice would only have 1 element then return a slice a step of 1 from start
            if num_steps == 1:
                return slice(start, start + 1, 1)
            # Ensure stop will be on a step from start but not greater than the length of the slice
            stop = min((start + (num_steps * step)), len(self))
            # Handle step greater than length of slice
            len_slice = abs(stop - start)
            if abs(step) >= len_slice:
                return slice(start, start + 1, 1)
            # Assert that the length of the slice is not 0 or negative or greater than the length of the tensor
            assert (len_slice > 0) and (len_slice <= len(self))
            # Assert that the range of the slice is valid
            assert (start >= 0) and (start <= len(self))
            if not (stop == (len(self) - 1)):
                assert (stop >= 0) and (stop <= len(self))
        except AssertionError:
            raise ValueError("Invalid slice with start, stop, step:", start, stop, step)
        return slice(start, stop, step)

    # Set the slices of the viewable tensor
    def set_slices(self, slices: List[slice]):
        for i, this_slice in enumerate(slices):
            # Populate none values in the slices and sanitize the slices
            slices[i] = self.simplify_slice(this_slice)
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
            key = slice(key, key + 1, 1)
        # Make sure the key range is valid
        self.simplify_slice(key)
        # Get the value
        real_index = 0
        contiguous_array = np.zeros(len(self), dtype=self.dtype)
        for this_slice in self.slices:
            # Get the start, stop, and step
            start = this_slice.start
            stop = this_slice.stop
            step = this_slice.step
            for slice_index in range((stop - start) // step):
                slice_index_real = (slice_index * step) + start
                # Set the slice
                contiguous_array[real_index] = self.value[slice_index_real]
                # Increment the real_index
                real_index += 1
        # Get the value slice
        value_slice = contiguous_array[key]
        # Length of the slice
        length = len(value_slice)
        # Shape is now flattened
        shape_slice = (length,)
        # Create the tensor
        tensor_slice = self.tensor_type(
            shape=shape_slice,
            dtype=self.dtype,
            value=value_slice,
            requires_grad=self.requires_grad,
        )
        # Return the tensor
        return tensor_slice

    def get_contiguous(self) -> "Tensor":
        # Return the contiguous tensor
        return self[0 : len(self)]

    # Redefine the setitem method to be symbolic
    def __setitem__(
        self,
        key: Union[int, slice],
        value: Union[List, np.ndarray, "Tensor", np.number, int, float],
    ):
        if not self.allow_modify:
            raise Exception("Modify not allowed")
        # If the key is an integer
        if type(key) is int:
            # Make into a slice
            key = slice(key, key + 1, 1)
        # Make sure the key range is valid
        self.simplify_slice(key)
        # Get the value
        real_index = 0
        contiguous_array = np.zeros(len(self), dtype=self.dtype)
        for this_slice in self.slices:
            # Get the start, stop, and step
            start = this_slice.start
            stop = this_slice.stop
            step = this_slice.step
            for slice_index in range((stop - start) // step):
                slice_index_real = (slice_index * step) + start
                # Set the slice
                contiguous_array[real_index] = self.value[slice_index_real]
                # Increment the real_index
                real_index += 1
        # Set the value
        contiguous_array[key] = value
        # Set the value
        self.value = contiguous_array

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
    def transpose(self, axes: Optional[List[int]] = None):
        # If the axes are not specified then set it to the reverse of first axis
        if axes is None:
            axes = list(range(len(self.viewable_shape) - 1, -1, -1))
        # Perform the transpose by modifying the slices
        # This needs to take into account the original shape of the viewable tensor and
        # the new shape of the viewable tensor to get the correct slices
        # Slices can also be any slice of the underlying tensor in any order and with overlaps or gaps
        # Slices can also have steps
        # Create a matrix of the indices to be used for the transpose
        matrix_post = np.zeros(len(self), dtype=np.int32)
        # Populate the matrix_pre with the slices
        matrix_index = 0
        for this_slice in self.slices:
            # Get the start, stop, and step
            start = this_slice.start
            stop = this_slice.stop
            step = this_slice.step
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
        self.viewable_shape = tuple(self.viewable_shape[i] for i in axes)


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

        test_arr = viewable_tensor.value.tolist()

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

        # Define a method to test that the test expressions are equal on normal lists
        # and then that our simplify_slice method returns the same result
        def test_simplify_slice(
            arr: List[int], test_slice: slice, expected_slice: Optional[slice] = None
        ):
            # Save a deepcopy of the input array
            arr_cp = arr[:]
            test_simplified_slice = viewable_tensor.simplify_slice(test_slice)
            print(test_slice, test_simplified_slice, expected_slice)
            self.assertEqual(arr[test_slice], arr[test_simplified_slice])
            if expected_slice is not None:
                self.assertEqual(arr[test_slice], arr[expected_slice])
                self.assertEqual(test_simplified_slice, expected_slice)
            # Assert that the input array is not modified
            self.assertEqual(arr, arr_cp)

        try:
            # Test the simplify_slice method
            # Test some known slice results
            test_simplify_slice(test_arr, slice(0, 4, 1), slice(0, 4, 1))
            test_simplify_slice(test_arr, slice(0, 0, 1), slice(0, 0, 1))
            test_simplify_slice(test_arr, slice(0, 1, 1), slice(0, 1, 1))
            test_simplify_slice(test_arr, slice(0, 2, -1), slice(0, 0, 1))
            test_simplify_slice(test_arr, slice(0, 1, 2), slice(0, 1, 1))
            test_simplify_slice(test_arr, slice(0, 2, 2), slice(0, 1, 1))
            test_simplify_slice(test_arr, slice(0, 3, 2), slice(0, 4, 2))
            test_simplify_slice(test_arr, slice(0, 1, -1), slice(0, 0, 1))
        except Exception as e:
            self.fail("Exception raised while testing simplify_slice method: " + str(e))

        try:
            # Test the simplify_slice method
            # Test random slices
            print("Testing random slices...")
            test_num = 100
            random_key = None
            for _ in range(test_num):
                start = np.random.randint(-8, 8)
                stop = np.random.randint(-8, 8)
                step = np.random.randint(-8, 8)
                if step == 0:
                    step = 1
                random_key = slice(start, stop, step)
                test_simplify_slice(test_arr, random_key)
            print("Finished testing random slices")
        except Exception as e:
            self.fail(
                "Exception raised while testing simplify_slice method with key({}): ".format(
                    random_key
                )
                + str(e)
            )

        # Define a method to test getitem
        def test_getitem(
            arr: List[int], viewable_tensor: ViewableTensor, key: Union[int, slice]
        ):
            # Check for equality
            self.assertEqual(viewable_tensor.value.tolist(), arr)
            # Get the value
            value_arr = arr[key]
            if type(key) is int:
                value_arr = [value_arr]
            value_tensor = viewable_tensor[key].value.tolist()
            # Test the value
            self.assertEqual(value_tensor, value_arr)

        try:
            # Perform random getitem tests
            print("Testing random int getitem...")
            test_num = 100
            random_key = None
            for _ in range(test_num):
                random_key = np.random.randint(0, 4)
                test_getitem(test_arr, viewable_tensor, random_key)
            print("Finished testing random int getitem")
            print("Testing random slice getitem...")
            for _ in range(test_num):
                start = np.random.randint(-8, 8)
                stop = np.random.randint(-8, 8)
                step = np.random.randint(-8, 8)
                if step == 0:
                    step = 1
                random_key = slice(start, stop, step)
                test_getitem(test_arr, viewable_tensor, random_key)
        except Exception as e:
            self.fail(
                "Exception raised while testing getitem method with key({}): ".format(
                    random_key
                )
                + str(e)
            )

        # Define a test to modify an input array and then test that the viewable tensor
        # is equal to the input array after the modification
        def test_setitem(
            arr: List[int],
            viewable_tensor: ViewableTensor,
            key: Union[int, slice],
            value: Union[int, List[int]],
        ):
            # Check for equality
            self.assertEqual(viewable_tensor.value.tolist(), arr)
            print(arr[key], key, value)
            # Set the value
            arr[key] = value
            viewable_tensor[key] = value
            # Test the value
            self.assertEqual(viewable_tensor.value.tolist(), arr)

        try:
            # Create a test array
            test_setitem_arr = [1, 2, 3, 4]
            # Test the setitem method
            print("Testing random key, value int pairs...")
            test_num = 100
            random_key = None
            random_value = None
            for _ in range(test_num):
                random_key = np.random.randint(0, 4)
                random_value = np.random.randint(0, 255)
                test_setitem(
                    test_setitem_arr, viewable_tensor, random_key, random_value
                )
            print("Finished testing random key, value int pairs")
            print("Testing random key, value slice pairs...")
            test_num = 100
            for _ in range(test_num):
                start = np.random.randint(-8, 8)
                stop = np.random.randint(-8, 8)
                step = np.random.randint(-8, 8)
                if step == 0:
                    step = 1
                random_key = slice(start, stop, step)
                sanitized_key = viewable_tensor.simplify_slice(random_key)
                length = abs(sanitized_key.stop - sanitized_key.start) // abs(
                    sanitized_key.step
                )
                if length == 0:
                    random_value = []
                else:
                    random_value = [np.random.randint(0, 255) for _ in range(length)]
                test_setitem(
                    test_setitem_arr, viewable_tensor, random_key, random_value
                )
            print("Finished testing random key, value slice pairs")
            # Reset the viewable tensor
            test_setitem(
                test_setitem_arr, viewable_tensor, slice(0, 4, 1), [1, 2, 3, 4]
            )
        except Exception as e:
            self.fail(
                "Exception raised while testing setitem method with key({}), value({}) and sanitized key({}): ".format(
                    random_key, random_value, viewable_tensor.simplify_slice(random_key)
                )
                + str(e)
            )

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


def unittest_viewable_tensor():
    print("Running unit tests for ViewableTensor...")
    # Run the unit tests
    unittest.main()
    print("Finished unit tests for ViewableTensor")


if __name__ == "__main__":
    unittest_viewable_tensor()
