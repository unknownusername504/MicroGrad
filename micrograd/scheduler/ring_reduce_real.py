import multiprocessing
import queue
from typing import Dict, List, get_args

import numpy as np

from micrograd.scheduler.schedule import CoreThreadAffinityManager
from micrograd.tensors.tensor import ScalarLike
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
        worker_contribution_array,
        num_workers,
        primitive_op_type,
        tolerate_remainder=True,
    ):
        # A loop is 1 iteration through this partial
        # We need 1 loop for the scatter-reduce, and 1 loop for the gather
        self.scatter_reduce_loop_complete = False

        self.worker_id = worker_id
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
            this_base_value = self.worker_contribution_array[start_index + index]
            assert isinstance(
                this_base_value, get_args(ScalarLike)
            ), "The base value is not a ScalarLike"
            self.worker_contribution_array[start_index + index] = (
                self.perform_primitive_op(
                    this_base_value, partial_result, is_reduce=True
                )
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
        return self.worker_contribution_array


class ModelRunner:
    def __init__(self, model, worker_id, timeout=100):
        self.model = model
        self.worker_id = worker_id
        self.core_thread_affinity = worker_id
        self.timeout = timeout

    @staticmethod
    def train_loop(model, core_thread_affinity, num_rounds, num_episodes_per_round):
        # Set the core affinity
        CoreThreadAffinityManager.set_thread_affinity(core_thread_affinity)
        for _ in range(num_rounds):
            # Send the model to the wave runner and train mini batches
            model.train(num_episodes_per_round)

    def get_worker_id(self):
        return self.worker_id

    def get_worker_contribution_array(self, num_workers):
        model_params = self.model.model.get_parameters()
        contributions = np.zeros(
            np.sum([np.prod(param.value.shape) for param in model_params])
        )
        # Divide the parameters by the number of Workers plus 1 for the base model
        index = 0
        for model_param in model_params:
            this_value = model_param.value / (num_workers + 1)
            contributions[index : index + np.prod(this_value.shape)] = (
                this_value.flatten()
            )
        return contributions

    def apply_updates(self, updates):
        self.model.model.set_parameters(updates)

    def run(self, num_rounds=2, num_episodes_per_round=100):
        # Start process
        process = multiprocessing.Process(
            target=ModelRunner.train_loop,
            args=(
                self.model,
                self.core_thread_affinity,
                num_rounds,
                num_episodes_per_round,
            ),
        )

        process.start()

        try:
            process.join(self.timeout)
        except TimeoutError:
            process.terminate()
            raise TimeoutError("The process timed out")


class EmuMultiWorker:
    # This class will use multiprocessing to emulate multiple Workers
    def __init__(self, ring_topology: RingTopology):
        self.num_workers = len(ring_topology.get_ring())
        self.processes: List[multiprocessing.Process] = [None] * self.num_workers
        # Create an input and output queue for each worker
        # Where the output queue of one worker is the input queue of the next worker
        self.worker_queues: List[Dict[str, multiprocessing.Queue]] = []
        self.manager = multiprocessing.Manager()
        # Initialize the queues
        for i in range(self.num_workers):
            self.worker_queues.append(
                {
                    "input_queue": None,
                    "output_queue": multiprocessing.Queue(),
                    "final_result": self.manager.list(),
                    "is_done": multiprocessing.Event(),
                }
            )
        for i in range(self.num_workers):
            # Set the input queue to previous worker's output queue
            input_worker_id = ring_topology.get_previous_worker(i)
            # debug_print(f"Worker {i} input queue: {input_worker_id}")
            this_input_queue = self.worker_queues[input_worker_id]["output_queue"]
            self.worker_queues[i]["input_queue"] = this_input_queue

    def create_worker(
        self,
        model_runner,
        num_rounds,
        num_episodes_per_round,
    ):
        # Create a process for worker and pass in the input and output queues
        worker_id = model_runner.get_worker_id()
        process = multiprocessing.Process(
            target=self.worker,
            args=(
                self.num_workers,
                model_runner,
                num_rounds,
                num_episodes_per_round,
                self.worker_queues[worker_id],
                DebugPrint.debug,
            ),
        )
        self.processes[worker_id] = process

    def start(self):
        for process in self.processes:
            if process is not None:
                # debug_print(f"Starting process {process.name}")
                process.start()

    def stop(self):
        for process in self.processes:
            if process is not None:
                # debug_print(f"Terminating process {process.name}")
                process.terminate()

    @staticmethod
    def worker(
        num_workers,
        model_runner,
        num_rounds,
        num_episodes_per_round,
        worker_queue,
        debug,
    ):
        # Carry over the debug mode to the process since it's not shared
        DebugPrint.set_debug(debug)

        # Perform work
        model_runner.run(num_rounds, num_episodes_per_round)

        # debug_print(f"Worker {multiprocessing.current_process().name} started")
        input_queue: multiprocessing.Queue = worker_queue["input_queue"]
        output_queue: multiprocessing.Queue = worker_queue["output_queue"]
        final_result: multiprocessing.Manager.ListProxy = worker_queue["final_result"]
        is_done: multiprocessing.Event = worker_queue["is_done"]

        worker_id = model_runner.get_worker_id()
        worker_contribution_array = model_runner.get_worker_contribution_array(
            num_workers
        )
        remainder = len(worker_contribution_array) % num_workers
        padding = (num_workers - remainder) if remainder != 0 else 0
        # Pad with zeros to the right if the remainder is not zero
        if padding != 0:
            worker_contribution_array = np.pad(
                worker_contribution_array,
                (0, padding),
                "constant",
                constant_values=(0),
            )
        operation = Operation(
            worker_id,
            worker_contribution_array,
            num_workers,
            "add",
        )
        # Prime the worker with the initial result
        data = operation.get_result()
        worker_queue["output_queue"].put(data)

        to_push_final_result = None
        while to_push_final_result is None:
            try:
                # debug_print(f"Worker {multiprocessing.current_process().name} waiting for data")
                data = input_queue.get(block=True, timeout=10)
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
                        output_queue.put(result, block=True, timeout=10)
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
                # Remove the padding
                final_result[:] = to_push_final_result[:-padding]
        # debug_print(f"Worker {multiprocessing.current_process().name} is done")
        is_done.set()

    # This method will be used to synchronize the workers
    def global_barrier(self):
        for worker_id in range(self.num_workers):
            worker_queue = self.worker_queues[worker_id]
            input_queue: multiprocessing.Queue = worker_queue["input_queue"]
            output_queue: multiprocessing.Queue = worker_queue["output_queue"]
            final_result: multiprocessing.Manager.ListProxy = worker_queue[
                "final_result"
            ]
            is_done: multiprocessing.Event = worker_queue["is_done"]
            # If the worker is not done, throw an error
            if not is_done.is_set():
                raise ValueError("The worker is not done")

            # Clear the queues
            while not input_queue.empty():
                input_queue.get()
            while not output_queue.empty():
                output_queue.get()
            final_result[:] = []
            is_done.clear()

            process = self.processes[worker_id]
            process.join()

    def gather_results(self):
        # Wait for all workers to finish
        for queue in self.worker_queues:
            is_done = queue["is_done"]
            is_done.wait()
        results = []
        for queue in self.worker_queues:
            final_result = queue["final_result"]
            # Convert the list proxy to a list
            final_result = list(final_result)
            results.append(final_result)
        self.global_barrier()
        return results


class RingCoordinator:
    def __init__(self, num_workers, base_model):
        self.num_workers = num_workers
        self.ring_topology = RingTopology(num_workers)
        self.base_model = base_model
        # Create an empty array to store the results
        self.emu_multi_worker = EmuMultiWorker(self.ring_topology)
        self.setup_ring()

    def setup_ring(self):
        self.model_runners = []
        for worker_id in range(self.num_workers):
            model = type(self.base_model)(do_load_model=True, do_save_model=False)
            # model.model.set_parameters(self.base_model.model.get_parameters())
            model_runner = ModelRunner(model, worker_id)
            self.model_runners.append(model_runner)

    def collapse_results(self, results):
        # Check if the results contain errors
        exceptions = []
        assert len(results) == self.num_workers, "The number of results is not correct"
        for result in results:
            if result is None:
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
        # Collapse the results into a single array
        collapsed_result = results[0]
        # Assert that the results are the same as each other
        for index in range(1, self.num_workers):
            this_result = results[index]
            assert this_result == collapsed_result, "The results are not the same"

        # Unflatten the results to match the list of parameters' shapes
        model_params = self.base_model.model.get_parameters()
        model_param_shapes = [param.shape for param in model_params]
        len_flattened_model_params = sum(np.prod(shape) for shape in model_param_shapes)
        assert (
            len(collapsed_result) == len_flattened_model_params
        ), "The result is not the correct length"
        for model_param_shape in model_param_shapes:
            this_param_len = np.prod(model_param_shape)
            this_param = collapsed_result[:this_param_len]
            this_param = np.array(this_param).reshape(model_param_shape)
            collapsed_result = collapsed_result[this_param_len:]

        return collapsed_result

    def reset_ring(self, num_rounds, num_episodes_per_round):
        for worker_id in range(self.num_workers):
            model_runner = self.model_runners[worker_id]
            self.emu_multi_worker.create_worker(
                model_runner,
                num_rounds,
                num_episodes_per_round,
            )

    def run_ring(self, num_rounds, num_episodes_per_round):
        self.reset_ring(num_rounds, num_episodes_per_round)
        # debug_print("Running the ring")
        self.emu_multi_worker.start()
        results = self.emu_multi_worker.gather_results()
        result = self.collapse_results(results)
        self.base_model.model.set_parameters(result)
        for model_runner in self.model_runners:
            model_runner.apply_updates(result)

    def stop_ring(self):
        self.emu_multi_worker.stop()
