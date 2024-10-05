# Class for viewing non contiguous tensor slices without copying

import copy
from typing import List, Optional, Union, Tuple
import unittest

import numpy as np
from micrograd.tensors.tensor import Tensor
from micrograd.utils.debug_utils import debug_print
from micrograd.utils.types import is_scalar_like, ScalarLike, TensorLike


class View:
    # TODO: Add a tile_skip_by so that tiles don't have to be contiguous
    # NOTE: Number of steps is a little odd as the first element is step 1 as in you took your first step into the view
    def __init__(
        self, start: int, num_steps: int, step_stride: int = 1, step_footprint: int = 1
    ):
        assert start >= 0
        assert num_steps > 0
        # if num_steps == 1 and step_footprint == 1:
        #    # Simplify step_stride to 1 if there is only 1 element
        #    step_stride = 1
        assert step_stride > 0
        assert step_footprint > 0
        # Convert complete tiling to a step size of 1 and take steps across the tiling instead
        if (step_footprint > 1) and (step_footprint == step_stride):
            num_steps *= step_stride
            step_footprint = 1
            step_stride = 1
        assert (
            step_footprint < step_stride
        ) or step_stride == 1, f"step_footprint={step_footprint} must be less than step_stride={step_stride} or step_stride must be 1"
        self.start = start
        self.num_steps = num_steps
        self.step_stride = step_stride
        self.step_footprint = step_footprint

    def __eq__(self, view):
        return (
            (self.start == view.start)
            and (self.num_steps == view.num_steps)
            and (self.step_stride == view.step_stride)
            and (self.step_footprint == view.step_footprint)
        )

    def __neq__(self, view):
        return not self.__eq__(view)

    def __hash__(self):
        return hash((self.start, self.num_steps, self.step_stride, self.step_footprint))

    def __str__(self):
        return f"View(start={self.start}, num_steps={self.num_steps}, step_stride={self.step_stride}, step_footprint={self.step_footprint})"

    # Define method for printing
    def __repr__(self):
        return self.__str__()

    # Define the length of the view
    def __len__(self):
        return self.num_steps * self.step_footprint

    def get_stop(self):
        return self.start + ((self.num_steps - 1) * self.step_stride)

    def to_slices(self):
        slices = []
        for i in range(self.step_footprint):
            start = self.start + i
            # Slice stop is not inclusive of exact multiple of step_stride
            stop = self.get_stop() + self.step_stride + i
            slices.append(slice(start, stop, self.step_stride))
        return slices

    # Convert a view to a list of indices
    def to_indices(self):
        indices = []
        for i in range(self.num_steps):
            for j in range(self.step_footprint):
                indices.append(self.start + (i * self.step_stride) + j)
        return indices

    @staticmethod
    def view_from_slice(slice: slice) -> "View":
        return View(slice.start, (slice.stop - slice.start) // slice.step, slice.step)

    @staticmethod
    def merge_views(
        view1: "View", view2: "View", allow_reorder: bool = True
    ) -> Union[Tuple["View"], Tuple["View", "View"]]:
        if view1 == view2:
            return tuple([view1])

        if view1.start > view2.start:
            if not allow_reorder:
                return tuple([view1, view2])
            view1, view2 = view2, view1
        elif view1.start == view2.start:
            if view2.get_stop() > view1.get_stop():
                view1, view2 = view2, view1

        stop_of_view1 = view1.get_stop()
        stop_of_view2 = view2.get_stop()
        is_view2_contained_in_view1 = (
            (view2.start >= view1.start)
            and (stop_of_view2 <= stop_of_view1)
            and (view2.step_footprint <= view1.step_footprint)
        )

        # Numbers of tiles must be the same or all elements of the smaller tiling must be contained in the larger tiling
        if view1.step_footprint != view2.step_footprint:
            if not is_view2_contained_in_view1:
                return tuple([view1, view2])

        # If view2 is contained in view1 and
        # step size of view2 is a multiple of step size of view1 (implying view2 granularity is <= view1) and
        # and start of view2 is a multiple of step size of view1 from view1's start,
        # then they are overlapping containment
        # TODO: You can be overlapped if you don't exactly land on a step in view2 but land on a tile and satisfy shifted containment conditions
        is_view2_overlapped_by_view1 = (
            is_view2_contained_in_view1
            and (view2.step_stride % view1.step_stride == 0)
            and ((view1.start + view2.start) % view1.step_stride == 0)
        )

        new_step_stride = max(
            view1.step_stride if (view1.num_steps != 1) else 1,
            view2.step_stride if (view2.num_steps != 1) else 1,
        )
        assert (
            (new_step_stride == 1)
            or (new_step_stride == view1.step_stride)
            or (new_step_stride == view2.step_stride)
        )
        if view1.step_stride != view2.step_stride:
            # 1 step can be transformed into any step size
            if (view1.num_steps != 1) and (view2.num_steps != 1):
                # Overlapping views can be merged
                if not is_view2_overlapped_by_view1:
                    return tuple([view1, view2])

        max_step = min(
            ((stop_of_view2 - view1.start) // new_step_stride) + 1, view1.num_steps + 1
        )
        assert max_step >= 1

        # Go through every step in view1 and attempt to land on a step in view2 with the next step
        # Also include 0 in case view2 is contained in view1 and they share the same start
        for step in range(0 if is_view2_contained_in_view1 else 1, max_step):
            this_step = view1.start + (step * new_step_stride)
            offset_in_view2 = this_step - view2.start
            if offset_in_view2 < 0:
                continue
            assert offset_in_view2 <= stop_of_view2
            if offset_in_view2 % view2.step_stride == 0:
                # Found a match
                new_start = view1.start
                new_stop = max(stop_of_view1, stop_of_view2)
                assert new_stop >= new_start
                assert (
                    (new_stop - new_start) % new_step_stride
                ) == 0, f"({new_stop} - {new_start}) % {new_step_stride} = {(new_stop - new_start) % new_step_stride} for {view1} and {view2}"
                new_num_steps = (
                    (max(stop_of_view1, stop_of_view2) - new_start) // new_step_stride
                ) + 1
                assert (new_num_steps > min(view1.num_steps, view2.num_steps)) and (
                    new_num_steps <= (view1.num_steps + view2.num_steps)
                ), f"{new_num_steps} would gain or lose elements from {view1} and {view2}"
                new_step_footprint = view1.step_footprint
                return (
                    View(new_start, new_num_steps, new_step_stride, new_step_footprint),
                )

        return tuple([view1, view2])

    def __add__(self, view):
        return View.merge_views(self, view)

    # Define random view generation
    @staticmethod
    def random_view(len_tensor, randomize_tiles: bool = False) -> "View":
        start = np.random.randint(0, len_tensor - 1)
        len_part = len_tensor - start
        num_steps = np.random.randint(1, len_part)
        # Don't allow step size to be so large that the steps are outside the tensor
        max_step_stride = len_part // num_steps
        if max_step_stride <= 1:
            step_stride = 1
        else:
            step_stride = np.random.randint(1, max_step_stride)
        if (step_stride > 2) and randomize_tiles:
            step_footprint = np.random.randint(1, step_stride - 1)
        else:
            step_footprint = 1
        return View(start, num_steps, step_stride, step_footprint)

    @staticmethod
    def test():
        # Create an array from 0 to 200
        index_array = np.arange(200)
        # rand_seed = int(time.time())
        rand_seed = 1
        np.random.seed(rand_seed)

        # Test merging with views
        test_tiling_merge = False
        found_merges = set()
        for _ in range(500):
            view1_params = {
                "start": np.random.randint(0, 10),
                "num_steps": np.random.randint(1, 10),
                "step_stride": np.random.randint(1, 10),
            }
            if test_tiling_merge and (view1_params["step_stride"] > 2):
                view1_params["step_footprint"] = np.random.randint(
                    1, view1_params["step_stride"] - 1
                )
            else:
                view1_params["step_footprint"] = 1
            view2_params = {
                "start": np.random.randint(0, 10),
                "num_steps": np.random.randint(1, 10),
                "step_stride": np.random.randint(1, 10),
            }
            if test_tiling_merge and (view2_params["step_stride"] > 2):
                view2_params["step_footprint"] = np.random.randint(
                    1, view2_params["step_stride"] - 1
                )
            else:
                view2_params["step_footprint"] = 1

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
                    for tile in range(view1.step_footprint):
                        combined_unmerged.append(index_array[view1_slices[tile]][step])
                view2_slices = view2.to_slices()
                for step in range(view2.num_steps):
                    for tile in range(view2.step_footprint):
                        combined_unmerged.append(index_array[view2_slices[tile]][step])
                # Sort and remove duplicates
                combined_unmerged = list(set(combined_unmerged))
                combined_unmerged.sort()
                debug_print(combined_unmerged)

                for merged_view in merged_views:
                    merged_slices = merged_view.to_slices()
                    for step in range(merged_view.num_steps):
                        for tile in range(merged_view.step_footprint):
                            combined_merged.append(
                                index_array[merged_slices[tile]][step]
                            )
                # Sort and remove duplicates
                combined_merged = list(set(combined_merged))
                combined_merged.sort()
                debug_print(combined_merged)
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
    def __init__(self, tensor: Tensor, views: Optional[List[View]] = None):
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

        # A list of views, which starts as the full tensor if not specified
        if views is None:
            self.views = [View(start=0, num_steps=len(self))]
        else:
            self.views = views
            self.merge_views()

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

    # Attempt to merge views to reduce the number of views
    def merge_views(self):
        # If there is only one view then return
        if len(self.views) == 1:
            return
        merge_index_outer = 0
        merged_views = []
        views = self.views
        view_is_merged = [0] * len(views)
        while merge_index_outer < len(views):
            if not view_is_merged[merge_index_outer]:
                view = views[merge_index_outer]
                view_is_merged[merge_index_outer] = 1
                for i, other_view in enumerate(views[merge_index_outer + 1 :]):
                    merge_index_inner = merge_index_outer + 1 + i
                    merged = View.merge_views(view, other_view, allow_reorder=False)
                    if len(merged) == 1:
                        view = merged[0]
                        view_is_merged[merge_index_inner] = 1
                merged_views.append(view)
            merge_index_outer += 1
        self.views = merged_views

    # Set the views of the viewable tensor
    def set_views(self, views: List[View]):
        self.views = views
        self.merge_views()

    # Redefine the len method to be symbolic
    def __len__(self):
        return np.prod(self.viewable_shape)

    # Get a list of indices for the viewable tensor
    def get_indices(self):
        indices = []
        for view in self.views:
            indices += view.to_indices()
        return indices

    # Redefine the getitem method to return a contiguous tensor
    def __getitem__(self, key: Union[int, View]) -> Tensor:
        indices = self.get_indices()
        # If the key is an integer then return the value at that index
        if isinstance(key, int):
            return self.tensor_type(
                shape=(1,), value=np.array([self.value[indices[key]]])
            )
        # If the key is a view then return the value at the view
        elif isinstance(key, View):
            slices = key.to_slices()
            combined = []
            for slice in slices:
                combined.extend(self.value[indices[slice]])
            return self.tensor_type(shape=(len(combined),), value=np.array(combined))
        else:
            raise Exception("Invalid key type")

    # Redefine the setitem method to be symbolic
    def __setitem__(
        self,
        key: Union[int, View],
        value: Union[TensorLike, ScalarLike],
    ):
        indices = self.get_indices()
        # If the key is an integer then set the value at that index
        if isinstance(key, int):
            # Check if the value is a tensor like
            if not is_scalar_like(value):
                assert len(value) == 1
                value = value[0]
            self.value[indices[key]] = value
        # If the key is a view then set the value at the view
        elif isinstance(key, View):
            if len(key) == 0:
                return

            slices = key.to_slices()
            if is_scalar_like(value):
                for this_slice in slices:
                    these_indices = indices[this_slice]
                    for index in these_indices:
                        self.value[index] = value
            else:
                value_index = 0
                for this_slice in slices:
                    these_indices = indices[this_slice]
                    len_these_indices = len(these_indices)
                    self.value[these_indices] = value[
                        value_index : value_index + len_these_indices
                    ]
                    value_index += len_these_indices
        else:
            raise Exception("Invalid key type")

    def get_contiguous(self) -> Tensor:
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

    # Redefine the transpose method to work with viewable tensors
    def transpose(self, axes: Optional[List[int]] = None):
        raise Exception("Transpose not implemented for ViewableTensor")


# Unit tests for ViewableTensor
class TestViewableTensor(unittest.TestCase):
    def test_viewable_tensor(self):
        try:
            # Create the tensor
            tensor = Tensor(np.array([[1, 2], [3, 4]], dtype=np.uint8))
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
        except Exception as e:
            self.fail("Exception raised while testing member variables: " + str(e))

        test_arr = viewable_tensor.value.tolist()

        try:
            # Test the flatten method
            viewable_tensor.flatten()
            self.assertEqual(viewable_tensor.viewable_shape, (4,))
        except Exception as e:
            self.fail("Exception raised while testing flatten method: " + str(e))

        try:
            # Test the reshape method
            viewable_tensor.reshape((2, 2))
            self.assertEqual(viewable_tensor.viewable_shape, (2, 2))
        except Exception as e:
            self.fail("Exception raised while testing reshape method: " + str(e))

        # Define a method to test view_length
        def test_view_length(arr: List[int], test_view: View):
            arr_from_view = []
            slices = test_view.to_slices()
            for slice in slices:
                arr_from_view += arr[slice]
            self.assertEqual(len(arr_from_view), len(test_view))

        try:
            # Test the view_length method
            # Test random views
            print("Testing random views for view_length method...")
            test_num = 100
            random_key = None
            for _ in range(test_num):
                random_key = View.random_view(len(test_arr))
                test_view_length(test_arr, random_key)
            print("Finished testing random views")
        except Exception as e:
            self.fail(
                "Exception raised while testing view_length method with key({}), sliced({}): ".format(
                    random_key, random_key.to_slices()
                )
                + str(e)
            )

        # Define a method to test getitem
        def test_getitem(
            arr: List[int], viewable_tensor: ViewableTensor, key: Union[int, View]
        ):
            # Check for equality
            self.assertEqual(viewable_tensor.value.tolist(), arr)
            # Get the value
            if type(key) is int:
                value_arr = [arr[key]]
            else:
                slices = key.to_slices()
                value_arr = []
                for slice in slices:
                    value_arr += arr[slice]
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
            print("Testing random view getitem...")
            for _ in range(test_num):
                random_key = View.random_view(len(test_arr))
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
            value: Union[ScalarLike, TensorLike],
        ):
            # Check for equality
            self.assertEqual(viewable_tensor.value.tolist(), arr)
            # Set the value
            if isinstance(key, int):
                arr[key] = value
                viewable_tensor[key] = value
            else:
                arr[key] = value
                as_view = View.view_from_slice(key)
                viewable_tensor[as_view] = value
            # Test the value
            self.assertEqual(viewable_tensor.value.tolist(), arr)

        try:
            # Create a test array
            test_setitem_arr = [1, 2, 3, 4]
            # Test the setitem method
            print("Testing random key, value int pairs for setitem method...")
            test_num = 100
            this_test_num = 0
            random_key = None
            random_value = None
            for _ in range(test_num):
                this_test_num += 1
                random_key = np.random.randint(0, 4)
                random_value = np.random.randint(0, 8)
                test_setitem(
                    test_setitem_arr, viewable_tensor, random_key, random_value
                )
            print("Finished testing random key, value int pairs")
            print("Testing random key, value view pairs for setitem method...")
            test_num = 100
            this_test_num = 0
            for _ in range(test_num):
                this_test_num += 1
                random_view = View.random_view(len(test_arr))
                for random_key in random_view.to_slices():
                    length = len(View.view_from_slice(random_key))
                    if length == 0:
                        random_value = []
                    else:
                        random_value = [np.random.randint(0, 8) for _ in range(length)]
                    test_setitem(
                        test_setitem_arr, viewable_tensor, random_key, random_value
                    )
            print("Finished testing random key, value view pairs")
            # Reset the viewable tensor
            viewable_tensor[View(start=0, num_steps=4)] = [1, 2, 3, 4]
        except Exception as e:
            self.fail(
                "Exception raised while testing setitem method at test number {} with key({}), value({}): ".format(
                    this_test_num, random_key, random_value
                )
                + str(e)
            )

        try:
            # Test the merge_views method
            print("Testing merge_views method...")
            viewable_tensor.set_views(
                [
                    View(start=0, num_steps=1),
                    View(start=1, num_steps=1),
                    View(start=2, num_steps=1),
                    View(start=3, num_steps=1),
                ]
            )
            self.assertEqual(viewable_tensor.views, [View(start=0, num_steps=4)])

            viewable_tensor.set_views(
                [
                    View(start=0, num_steps=2),
                    View(start=2, num_steps=2),
                ]
            )
            self.assertEqual(viewable_tensor.views, [View(start=0, num_steps=4)])

            viewable_tensor.set_views(
                [
                    View(start=2, num_steps=2),
                    View(start=0, num_steps=2),
                ]
            )
            self.assertEqual(
                viewable_tensor.views,
                [View(start=2, num_steps=2), View(start=0, num_steps=2)],
            )

            viewable_tensor.set_views(
                [
                    View(start=0, num_steps=4),
                ]
            )
            self.assertEqual(viewable_tensor.views, [View(start=0, num_steps=4)])

            viewable_tensor.merge_views()
            self.assertEqual(viewable_tensor.views, [View(start=0, num_steps=4)])
        except Exception as e:
            self.fail("Exception raised while testing merge_views method: " + str(e))

        # Reset the viewable tensor views
        viewable_tensor.set_views(
            [
                View(start=0, num_steps=4),
            ]
        )

        # TODO: Add more tests for ViewableTensor including transpose


def unittest_viewable_tensor():
    debug_print("Running unit tests for ViewableTensor...")
    # Run the unit tests
    unittest.main()
    print("Finished unit tests for ViewableTensor")


if __name__ == "__main__":
    # TODO: Move this to unittests
    View.test()
    unittest_viewable_tensor()
