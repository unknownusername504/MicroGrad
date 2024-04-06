# Class for viewing non contiguous tensor slices without copying

import copy
from typing import List, Optional, Union, Tuple
import unittest

import numpy as np
from micrograd.tensors.tensor import Tensor
from micrograd.utils.debug_utils import debug_print


class View:
    # TODO: Add a tile_skip_by so that tiles don't have to be contiguous
    # NOTE: Number of steps is a little odd as the first element is step 1 as in you took your first step into the view
    def __init__(
        self, start: int, num_steps: int, step_size: int = 1, num_tiles: int = 1
    ):
        assert start >= 0
        assert num_steps > 0
        assert step_size > 0
        assert num_tiles > 0
        # Convert complete tiling to a step size of 1 and take steps across the tiling instead
        if (num_tiles > 1) and (num_tiles == step_size):
            num_steps *= step_size
            num_tiles = 1
            step_size = 1
        assert (
            num_tiles < step_size
        ) or step_size == 1, f"num_tiles={num_tiles} must be less than step_size={step_size} or step_size must be 1"
        self.start = start
        self.num_steps = num_steps
        self.step_size = step_size
        self.num_tiles = num_tiles

    def __eq__(self, view):
        return (
            (self.start == view.start)
            and (self.num_steps == view.num_steps)
            and (self.step_size == view.step_size)
            and (self.num_tiles == view.num_tiles)
        )

    def __neq__(self, view):
        return not self.__eq__(view)

    def __hash__(self):
        return hash((self.start, self.num_steps, self.step_size, self.num_tiles))

    def __str__(self):
        return f"View(start={self.start}, num_steps={self.num_steps}, step_size={self.step_size}, num_tiles={self.num_tiles})"

    def get_stop(self):
        return self.start + ((self.num_steps - 1) * self.step_size)

    def to_slices(self):
        slices = []
        for i in range(self.num_tiles):
            start = self.start + i
            # Slice stop is not inclusive of exact multiple of step_size
            stop = self.get_stop() + self.step_size + i
            slices.append(slice(start, stop, self.step_size))
        return slices

    @staticmethod
    def merge_views(
        view1: "View", view2: "View"
    ) -> Union[Tuple["View"], Tuple["View", "View"]]:
        if view1 == view2:
            return tuple([view1])

        if view1.start > view2.start:
            view1, view2 = view2, view1
        elif view1.start == view2.start:
            if view2.get_stop() > view1.get_stop():
                view1, view2 = view2, view1

        stop_of_view1 = view1.get_stop()
        stop_of_view2 = view2.get_stop()
        is_view2_contained_in_view1 = (
            (view2.start >= view1.start)
            and (stop_of_view2 <= stop_of_view1)
            and (view2.num_tiles <= view1.num_tiles)
        )

        # Numbers of tiles must be the same or all elements of the smaller tiling must be contained in the larger tiling
        if view1.num_tiles != view2.num_tiles:
            if not is_view2_contained_in_view1:
                return tuple([view1, view2])

        # If view2 is contained in view1 and
        # step size of view2 is a multiple of step size of view1 (implying view2 granularity is <= view1) and
        # and start of view2 is a multiple of step size of view1 from view1's start,
        # then they are overlapping containment
        # TODO: You can be overlapped if you don't exactly land on a step in view2 but land on a tile and satisfy shifted containment conditions
        is_view2_overlapped_by_view1 = (
            is_view2_contained_in_view1
            and (view2.step_size % view1.step_size == 0)
            and ((view1.start + view2.start) % view1.step_size == 0)
        )

        new_step_size = max(
            view1.step_size if (view1.num_steps != 1) else 1,
            view2.step_size if (view2.num_steps != 1) else 1,
        )
        assert (
            (new_step_size == 1)
            or (new_step_size == view1.step_size)
            or (new_step_size == view2.step_size)
        )
        if view1.step_size != view2.step_size:
            # 1 step can be transformed into any step size
            if (view1.num_steps != 1) and (view2.num_steps != 1):
                # Overlapping views can be merged
                if not is_view2_overlapped_by_view1:
                    return tuple([view1, view2])

        max_step = min(
            max(((stop_of_view2 - view1.start) // new_step_size), 1), view1.num_steps
        )

        # Go through every step in view1 and attempt to land on a step in view2 with the next step
        # Also include 0 in case view2 is contained in view1 and they share the same start
        for step in range(0 if is_view2_contained_in_view1 else 1, max_step):
            this_step = view1.start + (step * new_step_size)
            offset_in_view2 = this_step - view2.start
            if offset_in_view2 < 0:
                continue
            assert offset_in_view2 <= stop_of_view2
            if offset_in_view2 % view2.step_size == 0:
                # Found a match
                new_start = view1.start
                new_stop = max(stop_of_view1, stop_of_view2)
                assert new_stop >= new_start
                assert (
                    (new_stop - new_start) % new_step_size
                ) == 0, f"({new_stop} - {new_start}) % {new_step_size} = {(new_stop - new_start) % new_step_size} for {view1} and {view2}"
                new_num_steps = (
                    (max(stop_of_view1, stop_of_view2) - new_start) // new_step_size
                ) + 1
                assert (new_num_steps > min(view1.num_steps, view2.num_steps)) and (
                    new_num_steps <= (view1.num_steps + view2.num_steps)
                ), f"{new_num_steps} would gain or lose elements from {view1} and {view2}"
                new_num_tiles = view1.num_tiles
                return (View(new_start, new_num_steps, new_step_size, new_num_tiles),)

        return tuple([view1, view2])

    def __add__(self, view):
        return View.merge_views(self, view)

    @staticmethod
    def test():
        # Create an array from 0 to 200
        index_array = np.arange(200)
        # rand_seed = int(time.time())
        rand_seed = 1
        np.random.seed(rand_seed)

        """
        # Test the view class
        for _ in range(5):
            start = np.random.randint(0, 10)
            num_steps = np.random.randint(1, 10)
            step_size = np.random.randint(1, 10)
            if step_size > 2:
                num_tiles = np.random.randint(1, step_size - 1)
            else:
                num_tiles = 1
            view = View(start, num_steps, step_size, num_tiles)
            debug_print(view)
            slices = view.to_slices()
            debug_print(slices)
            # Print the values of the array that are selected by the slices
            combined = []
            for step in range(num_steps):
                for tile in range(num_tiles):
                    combined.append(index_array[slices[tile]][step])
            debug_print(combined)
        """

        # Test merging with views
        test_tiling_merge = False
        found_merges = set()
        for _ in range(500):
            view1_params = {
                "start": np.random.randint(0, 10),
                "num_steps": np.random.randint(1, 10),
                "step_size": np.random.randint(1, 10),
            }
            if test_tiling_merge and (view1_params["step_size"] > 2):
                view1_params["num_tiles"] = np.random.randint(
                    1, view1_params["step_size"] - 1
                )
            else:
                view1_params["num_tiles"] = 1
            view2_params = {
                "start": np.random.randint(0, 10),
                "num_steps": np.random.randint(1, 10),
                "step_size": np.random.randint(1, 10),
            }
            if test_tiling_merge and (view2_params["step_size"] > 2):
                view2_params["num_tiles"] = np.random.randint(
                    1, view2_params["step_size"] - 1
                )
            else:
                view2_params["num_tiles"] = 1

            view1 = View(**view1_params)
            view2 = View(**view2_params)
            try:
                debug_print("view1: {}".format(view1))
                debug_print("view2: {}".format(view2))
                merged_views = view1 + view2
                for i, merged_view in enumerate(merged_views):
                    debug_print("merged_view{}: {}".format(i, merged_view))
                assert (len(merged_views) > 0) and (len(merged_views) <= 2)
                if len(merged_views) == 1:
                    debug_print("!!! Merged views !!!")
                    found_merges.add((view1, view2))

                combined_unmerged = []
                combined_merged = []

                view1_slices = view1.to_slices()
                for step in range(view1.num_steps):
                    for tile in range(view1.num_tiles):
                        combined_unmerged.append(index_array[view1_slices[tile]][step])
                view2_slices = view2.to_slices()
                for step in range(view2.num_steps):
                    for tile in range(view2.num_tiles):
                        combined_unmerged.append(index_array[view2_slices[tile]][step])
                # Sort and remove duplicates
                combined_unmerged = list(set(combined_unmerged))
                combined_unmerged.sort()
                # print(combined_unmerged)

                for merged_view in merged_views:
                    merged_slices = merged_view.to_slices()
                    for step in range(merged_view.num_steps):
                        for tile in range(merged_view.num_tiles):
                            combined_merged.append(
                                index_array[merged_slices[tile]][step]
                            )
                # Sort and remove duplicates
                combined_merged = list(set(combined_merged))
                combined_merged.sort()
                # print(combined_merged)
                assert (
                    combined_unmerged == combined_merged
                ), f"combined_unmerged: {combined_unmerged} != combined_merged: {combined_merged}"
            except AssertionError as e:
                print(f"Failed with view1: {view1}, view2: {view2}")
                raise e
        # Save the known merges to a file for later testing
        print(f"Found {len(found_merges)} merges")
        with open("known_merges.txt", "w") as f:
            f.write("[")
            for view1, view2 in found_merges:
                f.write(f"({view1},{view2}),")
            f.write("]")


class ViewableTensor(Tensor):
    def __init__(self, tensor: Tensor, slices: Optional[List[slice]] = None):
        self.tensor_type = type(tensor)
        self.viewable_shape = tensor.shape
        # Get the flattened tensor
        tensor.flatten()
        # Initialize the super class with the tensor's attributes
        super().__init__(
            shape=tensor.shape,
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

    @staticmethod
    def view_from_slice(slice: slice) -> View:
        return View(slice.start, (slice.stop - slice.start) // slice.step, slice.step)

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
        if self.slice_length(this_slice) == 0:
            simplified_slice = slice(0, 0, 1)
        else:
            # Get the start, stop, and step
            start = this_slice.start
            stop = this_slice.stop
            step = this_slice.step
            # If the step is none then set it to 1
            if step is None:
                step = 1
            # If the start is none then set it to 0
            if start is None:
                if step > 0:
                    start = 0
                else:
                    start = len(self) - 1
            # If the stop is none then set it to the length of the tensor
            if stop is None:
                if step > 0:
                    stop = len(self)
                else:
                    stop = -len(self) - 1
            # Clip the start and stop to the length of the tensor
            start = max(min(start, len(self)), -len(self))
            stop = max(min(stop, len(self)), (-len(self) - 1))
            simplified_slice = slice(start, stop, step)
        return simplified_slice

    def slice_length(self, this_slice: slice):
        return len(range(*this_slice.indices(len(self))))

    # Set the slices of the viewable tensor
    def set_slices(self, slices: List[slice]):
        for i, this_slice in enumerate(slices):
            # Populate none values in the slices
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
        new_slices = []
        # Iterate through the slices
        for index in range(0, len(self.slices)):
            next_slice = self.slices[index]
            slice_length = self.slice_length(next_slice)
            if slice_length == 0:
                # Skip empty slices
                continue
            # If slice arr is empty then add this slice and continue
            if len(new_slices) == 0:
                new_slices.append(next_slice)
                continue
            # Get the last slice
            last_slice = new_slices[-1]
            # If the slices have negative elements then they cannot be combined yet
            any_neg = (
                (last_slice.step < 0)
                or (next_slice.step < 0)
                or (last_slice.start < 0)
                or (next_slice.start < 0)
                or (last_slice.stop < 0)
                or (next_slice.stop < 0)
            )
            # Slices can be combined if they have the same step and the stop of the first is the start of the second
            # Or if the length of the next slice is 1
            if (
                not any_neg
                and (last_slice.stop == next_slice.start)
                and ((last_slice.step == next_slice.step) or (slice_length == 1))
            ):
                # Combine the slices
                new_slices[-1] = slice(
                    last_slice.start, next_slice.stop, last_slice.step
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
            slice_length = self.slice_length(this_slice)
            for slice_index in range(slice_length):
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
        self.slices = []
        for index in range(len(matrix_post)):
            # Get the index
            matrix_index = matrix_post[index]
            # Get the start, stop, and step
            start = matrix_index
            stop = matrix_index + 1
            step = 1
            # Create the slice
            this_slice = slice(start, stop, step)
            # Append the slice
            self.slices.append(this_slice)
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
                shape=(2, 2), value=np.array([[1, 2], [3, 4]], dtype=np.uint8)
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

        # Define a method to test slice_length
        def test_slice_length(arr: List[int], test_slice: slice):
            # Save a deepcopy of the input array
            arr_cp = arr[:]
            test_slice_length = viewable_tensor.slice_length(test_slice)
            self.assertEqual(len(arr[test_slice]), test_slice_length)
            # Assert that the input array is not modified
            self.assertEqual(arr, arr_cp)

        try:
            # Test the slice_length method
            # Test random slices
            debug_print("Testing random slices for slice_length method...")
            test_num = 100
            random_key = None
            for _ in range(test_num):
                start = np.random.randint(-8, 8)
                stop = np.random.randint(-8, 8)
                step = np.random.randint(-8, 8)
                if step == 0:
                    step = 1
                # Replace with None sometimes
                if np.random.random() < 0.05:
                    if np.random.random() < 0.33:
                        start = None
                    elif np.random.random() < 0.5:
                        stop = None
                    else:
                        step = None
                random_key = slice(start, stop, step)
                test_slice_length(test_arr, random_key)
            debug_print("Finished testing random slices")
        except Exception as e:
            self.fail(
                "Exception raised while testing slice_length method with key({}): ".format(
                    random_key
                )
                + str(e)
            )

        # Define a method to test that the test expressions are equal on normal lists
        # and then that our simplify_slice method returns the same result
        def test_simplify_slice(arr: List[int], test_slice: slice):
            # Save a deepcopy of the input array
            arr_cp = arr[:]
            test_simplified_slice = viewable_tensor.simplify_slice(test_slice)
            self.assertEqual(arr[test_slice], arr[test_simplified_slice])
            # Assert that the input array is not modified
            self.assertEqual(arr, arr_cp)

        try:
            # Test the simplify_slice method
            # Test random slices
            debug_print("Testing random slices for simplify_slice method...")
            test_num = 100
            random_key = None
            for _ in range(test_num):
                start = np.random.randint(-8, 8)
                stop = np.random.randint(-8, 8)
                step = np.random.randint(-8, 8)
                if step == 0:
                    step = 1
                # Replace with None sometimes
                if np.random.random() < 0.05:
                    if np.random.random() < 0.33:
                        start = None
                    elif np.random.random() < 0.5:
                        stop = None
                    else:
                        step = None
                random_key = slice(start, stop, step)
                test_simplify_slice(test_arr, random_key)
            debug_print("Finished testing random slices")
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
            debug_print("Testing random int getitem...")
            test_num = 100
            random_key = None
            for _ in range(test_num):
                random_key = np.random.randint(0, 4)
                test_getitem(test_arr, viewable_tensor, random_key)
            debug_print("Finished testing random int getitem")
            debug_print("Testing random slice getitem...")
            for _ in range(test_num):
                start = np.random.randint(-8, 8)
                stop = np.random.randint(-8, 8)
                step = np.random.randint(-8, 8)
                if step == 0:
                    step = 1
                # Replace with None sometimes
                if np.random.random() < 0.05:
                    if np.random.random() < 0.33:
                        start = None
                    elif np.random.random() < 0.5:
                        stop = None
                    else:
                        step = None
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
            # Set the value
            arr[key] = value
            viewable_tensor[key] = value
            # Test the value
            self.assertEqual(viewable_tensor.value.tolist(), arr)

        try:
            # Create a test array
            test_setitem_arr = [1, 2, 3, 4]
            # Test the setitem method
            debug_print("Testing random key, value int pairs for setitem method...")
            test_num = 100
            random_key = None
            random_value = None
            for _ in range(test_num):
                random_key = np.random.randint(0, 4)
                random_value = np.random.randint(0, 255)
                test_setitem(
                    test_setitem_arr, viewable_tensor, random_key, random_value
                )
            debug_print("Finished testing random key, value int pairs")
            debug_print("Testing random key, value slice pairs for setitem method...")
            test_num = 100
            for _ in range(test_num):
                start = np.random.randint(-8, 8)
                stop = np.random.randint(-8, 8)
                step = np.random.randint(-8, 8)
                if step == 0:
                    step = 1
                # Replace with None sometimes
                if np.random.random() < 0.05:
                    if np.random.random() < 0.33:
                        start = None
                    elif np.random.random() < 0.5:
                        stop = None
                    else:
                        step = None
                random_key = slice(start, stop, step)
                sanitized_key = viewable_tensor.simplify_slice(random_key)
                length = viewable_tensor.slice_length(sanitized_key)
                if length == 0:
                    random_value = []
                else:
                    random_value = [np.random.randint(0, 255) for _ in range(length)]
                test_setitem(
                    test_setitem_arr, viewable_tensor, random_key, random_value
                )
            debug_print("Finished testing random key, value slice pairs")
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
            debug_print("Testing combine_slices method...")
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

        # Method to test random slice combinations against a test array
        def test_combine_slices(viewable_tensor: ViewableTensor, slices: List[slice]):
            # Save a deepcopy of the input array
            arr = viewable_tensor.value.tolist()
            # Save the original slices
            original_slices = slices[:]
            # Combine the slices
            viewable_tensor.set_slices(slices)
            # Extract the slices
            slices = viewable_tensor.slices
            # Use the slices to get the value
            value_arr = []
            for this_slice in slices:
                value_arr += arr[this_slice]
            original_value_arr = []
            for this_slice in original_slices:
                original_value_arr += arr[this_slice]
            # Ensure the underlying array is not modified
            self.assertEqual(viewable_tensor.value.tolist(), arr)
            # Test the value
            self.assertEqual(value_arr, original_value_arr)

        try:
            # Test the combine_slices method
            debug_print(
                "Testing random slice combinations for combine_slices method..."
            )
            test_num = 100
            this_test_num = 0
            random_slices = None
            for _ in range(test_num):
                this_test_num += 1
                random_slice_1_start = np.random.randint(-8, 8)
                random_slice_1_stop = np.random.randint(-8, 8)
                random_slice_1_step = np.random.randint(-8, 8)
                if random_slice_1_step == 0:
                    random_slice_1_step = 1
                random_slice_2_start = np.random.randint(-8, 8)
                random_slice_2_stop = np.random.randint(-8, 8)
                random_slice_2_step = np.random.randint(-8, 8)
                if random_slice_2_step == 0:
                    random_slice_2_step = 1
                random_slice_1 = slice(
                    random_slice_1_start, random_slice_1_stop, random_slice_1_step
                )
                random_slice_2 = slice(
                    random_slice_2_start, random_slice_2_stop, random_slice_2_step
                )
                random_slices = [random_slice_1, random_slice_2]
                test_combine_slices(viewable_tensor, random_slices)
            debug_print("Finished testing random slice combinations")
        except Exception as e:
            self.fail(
                "Exception raised while testing at test num ({}) combine_slices method with slices({}): ".format(
                    this_test_num, random_slices
                )
                + str(e)
            )

        # Reset the viewable tensor slices
        viewable_tensor.set_slices(
            [
                slice(0, 4, 1),
            ]
        )

        # NOTE: Transpose does not actually work yet, just this one test case
        try:
            # Test the transpose method
            debug_print("Testing transpose method...")
            viewable_tensor.transpose()
            self.assertEqual(viewable_tensor.viewable_shape, (2, 2))
            self.assertEqual(viewable_tensor.value.tolist(), [1, 2, 3, 4])
            transposed_array = viewable_tensor.get_contiguous().value.tolist()
            # Transposed array will be [[1, 3], [2, 4]]
            self.assertEqual(transposed_array, [1, 3, 2, 4])
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
    debug_print("Running unit tests for ViewableTensor...")
    # Run the unit tests
    unittest.main()
    debug_print("Finished unit tests for ViewableTensor")


if __name__ == "__main__":
    unittest_viewable_tensor()
