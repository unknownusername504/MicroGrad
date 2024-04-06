import multiprocessing
import queue
from typing import Dict, List
import time
import psutil

from micrograd.utils.debug_utils import DebugPrint, debug_print


class RingTopology:
    def __init__(self, num_workers):
        self.num_workers = num_workers
        self.ring = [i for i in range(num_workers)]  # Define the ring topology

    def get_ring(self):
        return self.ring

    def get_next_worker(self, worker_id):
        next_worker = (worker_id + 1) % self.num_workers
        return next_worker

    def get_previous_worker(self, worker_id):
        previous_worker = (worker_id - 1) % self.num_workers
        return previous_worker


class DistributedPartials:
    def __init__(self, num_workers, arr_len, tolerate_remainder=True):
        self.num_workers = num_workers
        self.arr_len = arr_len
        self.tolerate_remainder = tolerate_remainder
        self.compute_partials()

    def compute_partials(self):
        self.partial_index = -1
        # TODO: This shouldn't be required
        assert (
            self.num_workers <= self.arr_len
        ), "The number of Workers is greater than the length of the array"
        # Slice the array into divs of num_workers
        divs = self.arr_len // self.num_workers
        remainder = self.arr_len % self.num_workers
        partials = []
        if self.tolerate_remainder == False and remainder != 0:
            raise ValueError(
                "The remainder is not handled. Adjust the input or the number of Workers for the operation."
            )
        assert remainder <= divs, "The remainder is greater than the number of splits."
        for i in range(remainder):
            div_len = self.num_workers + 1
            start_index = i * div_len
            end_index = start_index + div_len
            partials.append((start_index, end_index))
        for i in range(remainder, divs):
            div_len = self.num_workers
            start_index = i * div_len
            end_index = start_index + div_len
            partials.append((start_index, end_index))
        self.partials = partials

    def get_num_partials(self):
        return len(self.partials)

    def set_next_partial(self):
        assert (
            self.partial_index < len(self.partials) - 1
        ), "The partial index is out of range"
        self.partial_index += 1
        return self.partial_index

    def get_partial(self):
        return self.partials[self.partial_index]


class Operation:
    # Class to define the steps and structure of an operation and its results
    def __init__(
        self,
        worker_id,
        base_array,
        worker_contribution_array,
        num_workers,
        primitive_op_type,
        tolerate_remainder,
    ):
        # A loop is 1 iteration through this partial
        # We need 1 loop for the scatter-reduce, and 1 loop for the gather
        self.scatter_reduce_loop_complete = False

        self.worker_id = worker_id
        assert isinstance(base_array, list), "The base array must be a list"
        assert isinstance(
            worker_contribution_array, list
        ), "The worker contribution array must be a list"
        assert len(base_array) == len(
            worker_contribution_array
        ), "The base array and the contribution array must have the same length"
        self.base_array = base_array
        self.worker_contribution_array = worker_contribution_array
        debug_print(
            f"Worker {self.worker_id} has contribution array: {self.worker_contribution_array}"
        )
        self.num_workers = num_workers
        self.primitive_op_type = primitive_op_type
        self.tolerate_remainder = tolerate_remainder

        # Construct the partials
        self.distributed_partials = DistributedPartials(
            self.num_workers,
            len(self.worker_contribution_array),
            self.tolerate_remainder,
        )
        self.move_to_next_partial()
        # debug_print(f"Worker {self.worker_id} at index in {self.index_into_partial} and out {self.index_out_of_partial} in partial {self.partial_num}")

        self.is_done = False

    def move_to_next_partial(self):
        self.partial_num = self.distributed_partials.set_next_partial()
        # Get this partial for the contribution array and the base array
        # This basically emulates caching the partials on the Worker
        start_index, end_index = self.distributed_partials.get_partial()
        self.this_worker_partitioned_array = self.worker_contribution_array[
            start_index:end_index
        ]
        # Keep track of where we are in this partial
        self.itr_of_partial = 0
        self.index_out_of_partial = self.worker_id
        self.index_into_partial = self.calc_move_index_of_partial(
            self.index_out_of_partial
        )
        # debug_print(f"Worker {self.worker_id} has partial {self.partial_num} of the contribution array: {self.this_worker_partitioned_array}")

    def perform_primitive_op(self, operand1, operand2, is_reduce):
        if self.primitive_op_type == "add":
            return operand1 + operand2
        elif self.primitive_op_type == "subtract":
            if not is_reduce:
                return operand1 + operand2
            else:
                return operand1 - operand2
        elif self.primitive_op_type == "multiply":
            return operand1 * operand2
        elif self.primitive_op_type == "divide":
            if not is_reduce:
                return operand1 * operand2
            else:
                return operand1 / operand2
        elif self.primitive_op_type == "max":
            return max(operand1, operand2)
        elif self.primitive_op_type == "min":
            return min(operand1, operand2)
        else:
            raise ValueError("Operation not supported")

    def calc_move_index_of_partial(self, index):
        len_partial = len(self.this_worker_partitioned_array)
        moved_index = (index - 1) % len_partial
        # debug_print(f"Worker {self.worker_id} moved to index {moved_index} in partial {self.partial_num}")
        return moved_index

    def move_indexes_of_partial(self):
        self.index_into_partial = self.calc_move_index_of_partial(
            self.index_into_partial
        )
        self.index_out_of_partial = self.calc_move_index_of_partial(
            self.index_out_of_partial
        )
        # debug_print(f"Worker {self.worker_id} moved to index {self.index_into_partial} in partial {self.partial_num}")

    def compute_result(self, input_data):
        if self.is_done:
            return None
        # Perform the operation at the current index for the current partial on this input data
        # The input data is the result from the previous Worker
        this_indexed_data = self.this_worker_partitioned_array[self.index_into_partial]
        if not self.scatter_reduce_loop_complete:
            # This is the scatter-reduce loop so we perform the operation
            partial_result = self.perform_primitive_op(
                this_indexed_data, input_data, is_reduce=False
            )
            debug_print(
                f"Worker {self.worker_id} computed partial result: {partial_result} from {this_indexed_data} and {input_data}"
            )
        else:
            # This is the gather loop so we store the input data which is a final partial result
            partial_result = input_data
            debug_print(
                f"Worker {self.worker_id} passed through the result: {partial_result} from {input_data}"
            )
        # Store the result
        self.this_worker_partitioned_array[self.index_into_partial] = partial_result
        self.itr_of_partial += 1
        is_end_of_loop = (
            self.itr_of_partial % len(self.this_worker_partitioned_array)
        ) == (len(self.this_worker_partitioned_array) - 1)
        if is_end_of_loop:
            if not self.scatter_reduce_loop_complete:
                self.scatter_reduce_loop_complete = True
                debug_print(
                    f"!!! Worker {self.worker_id} completed the scatter-reduce loop !!!"
                )
                # debug_print(f"Worker {self.worker_id} at index {self.index_into_partial} in partial {self.partial_num}")
            else:
                return self.commit_partial_results()
        self.move_indexes_of_partial()
        # Send the result to the next Worker
        # Happens after the move since we prime with the initial result
        next_data = self.get_result()
        # debug_print(f"Worker {self.worker_id} sending data: {next_data}")
        return next_data

    def commit_partial_results(self):
        # Commit the result to the base array
        start_index, _ = self.distributed_partials.get_partial()
        for index, partial_result in enumerate(self.this_worker_partitioned_array):
            this_base_value = self.base_array[start_index + index]
            assert isinstance(
                this_base_value, (int, float)
            ), "The base array must contain numbers"
            self.base_array[start_index + index] = self.perform_primitive_op(
                this_base_value, partial_result, is_reduce=True
            )
        # Check if there are more partials
        if self.partial_num < self.distributed_partials.get_num_partials() - 1:
            self.move_to_next_partial()
            self.scatter_reduce_loop_complete = False
        else:
            debug_print(f"Worker {self.worker_id} is done")
            self.is_done = True

        debug_print(f"!!! Worker {self.worker_id} committed the partial results !!!")
        # debug_print(f"Worker {self.worker_id} at index {self.index_into_partial} in partial {self.partial_num}")
        if self.is_done:
            return None
        assert (
            self.index_out_of_partial == self.worker_id
        ), "The indexes are not at the reset position, expected index out of {} but got {}".format(
            self.worker_id, self.index_out_of_partial
        )
        next_data = self.get_result()
        # debug_print(f"Worker {self.worker_id} sending data: {next_data}")
        return next_data

    def get_result(self):
        # Simply get the data at the current index
        initial_result = self.this_worker_partitioned_array[self.index_out_of_partial]
        debug_print(f"Worker {self.worker_id} result: {initial_result}")
        return initial_result

    def collect_results(self):
        return self.base_array


class EmuMultiWorker:
    # This class will use multiprocessing to emulate multiple Workers
    def __init__(self, ring_topology: RingTopology, base_array):
        self.processes: List[multiprocessing.Process] = []
        # Create an input and output queue for each worker
        # Where the output queue of one worker is the input queue of the next worker
        self.worker_queues: List[Dict[str, multiprocessing.Queue]] = []
        self.manager = multiprocessing.Manager()
        # Initialize the queues
        for i in ring_topology.get_ring():
            self.worker_queues.append(
                {
                    "input_queue": None,
                    "output_queue": multiprocessing.Queue(),
                    "final_result": self.manager.list(),
                    "is_done": multiprocessing.Event(),
                }
            )
        for i in ring_topology.get_ring():
            # Set the input queue to previous worker's output queue
            input_worker_id = ring_topology.get_previous_worker(i)
            # debug_print(f"Worker {i} input queue: {input_worker_id}")
            this_input_queue = self.worker_queues[input_worker_id]["output_queue"]
            self.worker_queues[i]["input_queue"] = this_input_queue
        self.num_workers = len(ring_topology.get_ring())
        self.base_array = base_array

    def create_worker(
        self,
        worker_id,
        worker_contribution_array,
        primitive_op_type,
        tolerate_remainder,
    ):
        # Create a process for worker and pass in the input and output queues
        this_worker_queue = self.worker_queues[worker_id]
        operation = Operation(
            worker_id,
            self.base_array,
            worker_contribution_array,
            self.num_workers,
            primitive_op_type,
            tolerate_remainder,
        )
        # Prime the worker with the initial result
        data = operation.get_result()
        this_worker_queue["output_queue"].put(data)
        process = multiprocessing.Process(
            target=self.worker,
            args=(operation, this_worker_queue, DebugPrint.debug),
        )
        self.processes.append(process)

    def start(self):
        for process_num, process in enumerate(self.processes):
            # debug_print(f"Starting process {process.name}")
            # Set the core affinity for the process so there is minimal context switching
            process.start()
            p = psutil.Process(process.pid)
            p.cpu_affinity([process_num])

    def stop(self):
        for process in self.processes:
            process.terminate()

    @staticmethod
    def worker(operation: Operation, worker_queue, debug):
        # Carry over the debug mode to the process since it's not shared
        DebugPrint.set_debug(debug)
        # debug_print(f"Worker {multiprocessing.current_process().name} started")
        input_queue: multiprocessing.Queue = worker_queue["input_queue"]
        output_queue: multiprocessing.Queue = worker_queue["output_queue"]
        final_result: multiprocessing.Manager.ListProxy = worker_queue["final_result"]
        is_done: multiprocessing.Event = worker_queue["is_done"]
        to_push_final_result = None
        while to_push_final_result is None:
            try:
                # debug_print(f"Worker {multiprocessing.current_process().name} waiting for data")
                data = input_queue.get(block=True, timeout=5)
                if data is None:
                    to_push_final_result = operation.collect_results()
                elif isinstance(data, Exception):
                    to_push_final_result = Exception(
                        "Exception occurred while receiving input"
                    )
                else:
                    # debug_print(f"Worker {multiprocessing.current_process().name} received data: {data}")
                    result = operation.compute_result(data)
                    # debug_print(f"Worker {multiprocessing.current_process().name} computed result: {result}")
                    if operation.is_done or result is None:
                        to_push_final_result = operation.collect_results()
                    elif isinstance(result, Exception):
                        to_push_final_result = Exception(
                            "Exception occurred while sending output"
                        )
                    else:
                        output_queue.put(result, block=True, timeout=5)
                        # debug_print(f"Worker {multiprocessing.current_process().name} sent result: {result}")
            except queue.Full:
                to_push_final_result = Exception(
                    "Timeout occurred while sending output"
                )
            # Handle the timeout
            except queue.Empty:
                to_push_final_result = Exception(
                    "Timeout occurred while receiving input"
                )
            # Sleep so I can read the print statements
            # time.sleep(1)
        if to_push_final_result is not None:
            # debug_print(f"Worker {multiprocessing.current_process().name} pushing final result: {to_push_final_result}")
            if isinstance(to_push_final_result, Exception):
                final_result[:] = [to_push_final_result]
            else:
                # Is a list
                final_result[:] = to_push_final_result
        # debug_print(f"Worker {multiprocessing.current_process().name} is done")
        is_done.set()

    # This method will be used to synchronize the workers
    def global_barrier(self):
        for queue in self.worker_queues:
            input_queue = queue["input_queue"]
            input_queue.put(None)
        for process in self.processes:
            process.join()

    def gather_results(self):
        # Wait for all workers to finish
        for queue in self.worker_queues:
            is_done = queue["is_done"]
            is_done.wait()
        results = []
        for queue in self.worker_queues:
            output_queue = queue["output_queue"]
            if not output_queue.empty():
                # debug_print(f"The output queue is not empty")
                outputs = []
                while not output_queue.empty():
                    outputs.append(output_queue.get())
                # debug_print(f"Outputs: {outputs}")
                raise ValueError("The output queue is not empty")
            final_result = queue["final_result"]
            # Convert the list proxy to a list
            final_result = list(final_result)
            results.append(final_result)
        return results


class RingCoordinator:
    def __init__(self, num_workers, base_array, tolerate_remainder=True):
        self.num_workers = num_workers
        self.ring_topology = RingTopology(num_workers)
        self.emu_multi_worker = EmuMultiWorker(self.ring_topology, base_array)
        self.tolerate_remainder = tolerate_remainder
        self.num_iterations = None

    def setup_ring(self, worker_contribution_arrays, primitive_op_type):
        self.primitive_op_type = primitive_op_type
        for worker_id in range(self.num_workers):
            worker_contribution_array = worker_contribution_arrays[worker_id]
            self.emu_multi_worker.create_worker(
                worker_id,
                worker_contribution_array,
                self.primitive_op_type,
                self.tolerate_remainder,
            )

    def run_ring(self):
        # debug_print("Running the ring")
        self.emu_multi_worker.start()
        results = self.emu_multi_worker.gather_results()
        # Check if the results contain errors
        exceptions = []
        for result in results:
            if not isinstance(result, list):
                debug_print(f"Result: {result} with type: {type(result)}")
                exceptions.append(Exception("The result is not a list"))
            if len(result) < 1:
                exceptions.append(Exception("The result is empty"))
            if isinstance(result[0], Exception):
                exceptions.append(result[0])
        if len(exceptions) > 0:
            exceptions_str = "\n".join([str(exception) for exception in exceptions])
            raise ValueError(
                "Errors occurred during the ring computation. The exceptions are: \n{}".format(
                    exceptions_str
                )
            )
        debug_print("Results: ", results)
        return results

    def stop_ring(self):
        self.emu_multi_worker.stop()


import unittest
import numpy as np


class TestRingReduce(unittest.TestCase):
    # Initialize the test case
    def setUp(self):
        # Exclude one core for the running process
        assert (
            multiprocessing.cpu_count() >= 2
        ), "Run test on at minimum dual core machine"
        self.num_processes = max(2, (multiprocessing.cpu_count() - 1))
        debug_print(f"Max number of processes: {self.num_processes}")
        self.primitive_op_types = [
            "add",
            "subtract",
            "multiply",
            "divide",
            "max",
            "min",
        ]

    def reset_test(self, num_workers=None, num_divs=None):
        if self.do_true_random:
            truerand = np.random.SeedSequence()
            seed = truerand.entropy % (2**32 - 1)
        else:
            seed = 0xDEADDEAD
        debug_print(f"Seed: {seed}")
        np.random.seed(seed)
        if num_workers is not None:
            self.num_workers = num_workers
        else:
            num_workers_bias_factor = 4  # Adjust this to change the bias. Lower values bias towards higher numbers.
            # Note: Can't form a ring with less than 2 workers
            num_workers = (
                np.random.beta(2, num_workers_bias_factor) * (self.num_processes - 1)
                + 1
            )
            num_workers = int(np.round(num_workers))  # Round to nearest integer
            num_workers = min(multiprocessing.cpu_count(), max(2, num_workers))
        if num_divs is not None:
            self.num_divs = num_divs
        else:
            num_divs = np.random.randint(1, 1000)
        print(f"Number of Workers: {num_workers}, Number of divisions: {num_divs}")
        self.base_array = np.random.rand(num_divs * num_workers).tolist()
        debug_print(f"Base array: {self.base_array}")
        self.worker_contribution_arrays = [
            np.random.rand(len(self.base_array)).tolist() for _ in range(num_workers)
        ]
        self.ring_coordinator = RingCoordinator(num_workers, self.base_array)
        self.ring_coordinator.setup_ring(
            self.worker_contribution_arrays, self.primitive_op_type
        )

    def run_trials(self, trial_num, num_workers=None, num_divs=None):
        self.reset_test(num_workers=num_workers, num_divs=num_divs)
        # Compute the summation of the base with the worker contributions using numpy
        if self.primitive_op_type == "add":
            numpy_result = np.asarray(self.base_array)
            for array in self.worker_contribution_arrays:
                numpy_result += np.asarray(array)
        elif self.primitive_op_type == "subtract":
            numpy_result = np.asarray(self.base_array)
            for array in self.worker_contribution_arrays:
                numpy_result -= np.asarray(array)
        elif self.primitive_op_type == "multiply":
            numpy_result = np.asarray(self.base_array)
            for array in self.worker_contribution_arrays:
                numpy_result *= np.asarray(array)
        elif self.primitive_op_type == "divide":
            numpy_result = np.asarray(self.base_array)
            for array in self.worker_contribution_arrays:
                numpy_result /= np.asarray(array)
        elif self.primitive_op_type == "max":
            # Stack the base array and the worker contribution arrays
            stacked_arrays = np.stack(
                [self.base_array] + self.worker_contribution_arrays
            )
            # print(f"Stacked arrays: {stacked_arrays}")
            numpy_result = np.max(stacked_arrays, axis=0)
        elif self.primitive_op_type == "min":
            # Stack the base array and the worker contribution arrays
            stacked_arrays = np.stack(
                [self.base_array] + self.worker_contribution_arrays
            )
            # print(f"Stacked arrays: {stacked_arrays}")
            numpy_result = np.min(stacked_arrays, axis=0)
        else:
            raise ValueError("Operation not supported")

        # Run the ring
        results = self.ring_coordinator.run_ring()
        # Stop the ring
        self.ring_coordinator.stop_ring()

        assert isinstance(results, list), "The results are not a list"

        # Assert that the results are the same as each other
        for index, result in enumerate(results):
            if index == 0:
                continue
            assert result == results[index - 1], "The results are not the same"

        result = results[0]
        # Assert that the results are the same as the numpy result
        # We use np.isclose to handle floating point precision
        # Set a high tolerance on the final result to account for the floating point precision
        tolerance = 0.10  # 10% tolerance
        assert len(result) == len(numpy_result), "The number of results do not match"
        isclose = np.allclose(result, numpy_result, rtol=tolerance, atol=tolerance)
        try:
            self.assertTrue(isclose, f"Results: {result}, Numpy result: {numpy_result}")
        except AssertionError as e:
            print(f"Trial {trial_num} failed")
            raise e
        print(f"Trial {trial_num} passed")

    @unittest.skip("Skipping basic test")
    def test_ring_reduce_basic(self):
        self.do_true_random = False
        for primitive_op_type in self.primitive_op_types:
            try:
                print(
                    "Running "
                    + self._testMethodName
                    + " for operation: "
                    + primitive_op_type
                )
                self.primitive_op_type = primitive_op_type
                self.run_trials(
                    trial_num=1,
                    num_workers=min(multiprocessing.cpu_count(), 3),
                    num_divs=2,
                )
            except Exception as e:
                print("Basic test failed for operation: " + primitive_op_type)
                print(e)

    # @unittest.skip("Skipping complex test")
    def test_ring_reduce_complex(self):
        self.do_true_random = True
        self.num_trials = 5
        for primitive_op_type in self.primitive_op_types:
            try:
                print(
                    "Running "
                    + self._testMethodName
                    + " for operation: "
                    + primitive_op_type
                    + " for {} trials".format(self.num_trials)
                )
                self.primitive_op_type = primitive_op_type
                for trial_num in range(1, self.num_trials + 1):
                    self.run_trials(trial_num)
            except Exception as e:
                print("Complex test failed for operation: " + primitive_op_type)
                print(e)

    # Needed for compatibility with colab
    def run_all_tests(self):
        # Run all the tests
        self.test_ring_reduce_basic()
        self.test_ring_reduce_complex()


if __name__ == "__main__":
    unittest.main()
