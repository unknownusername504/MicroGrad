from typing import ClassVar
import multiprocessing
import os
from typing import List

from micrograd.tensors.tensor import Tensor


# Class representing a wave process output of none
class WaveProcessOutputNone:
    def __init__(self):
        pass


# Class with wave process structure
class WaveProcessJob:
    # Class with wave process job
    def __init__(self, func, args):
        self.func = func
        self.args = args


class WaveProcessWorker(multiprocessing.Process):
    # Leave one core free for the main process
    num_cores: ClassVar[int] = multiprocessing.cpu_count() - 1
    num_hyperthreads: ClassVar[int] = 2
    # Allow for hyperthreading
    num_threads_free: ClassVar[multiprocessing.Value] = multiprocessing.Value(
        "i", (num_cores * num_hyperthreads)
    )

    core_thread_acquire_affinity_lock: ClassVar[
        multiprocessing.Lock
    ] = multiprocessing.Lock()
    # Two locks for hyperthreading
    core_thread_affinity_locks: ClassVar[List[multiprocessing.Lock]] = [
        multiprocessing.Lock() for _ in range(num_cores * num_hyperthreads)
    ]

    # Create the input and output queues
    input_queue: ClassVar[multiprocessing.Queue] = multiprocessing.Queue()
    output_queue: ClassVar[multiprocessing.Queue] = multiprocessing.Queue()

    def enqueue(self, input):
        WaveProcessWorker.input_queue.put(input)

    def dequeue(self, timeout=10):
        return WaveProcessWorker.output_queue.get(timeout=timeout)

    def lock_core_thread_affinity(self):
        with WaveProcessWorker.core_thread_acquire_affinity_lock:
            # Find a core that is free
            for core_id in range(WaveProcessWorker.num_cores):
                for hyperthread_id in range(WaveProcessWorker.num_hyperthreads):
                    # Calculate the thread id
                    core_thread_id = (
                        core_id * WaveProcessWorker.num_hyperthreads + hyperthread_id
                    )
                    # Check the lock status
                    if not WaveProcessWorker.core_thread_affinity_locks[
                        core_thread_id
                    ].locked():
                        # Return the core id
                        WaveProcessWorker.num_threads_free.value -= 1
                        os.sched_setaffinity(0, core_id)
                        return core_thread_id
            # If no cores are free, raise an exception
            raise Exception("No cores are free")

    def unlock_core_thread_affinity(self, core_thread_id):
        with WaveProcessWorker.core_thread_acquire_affinity_lock:
            # Unlock the core
            WaveProcessWorker.num_threads_free.value += 1
            WaveProcessWorker.core_thread_affinity_locks[core_thread_id].release()

    class acquire_core_thread_affinity:
        def __init__(self):
            self.core_thread_id = None

        def __enter__(self):
            self.core_thread_id = WaveProcessWorker.lock_core_thread_affinity()

        def __exit__(self, exc_type, exc_value, traceback):
            WaveProcessWorker.unlock_core_thread_affinity(self.core_thread_id)

    def create_pool(self, chunks: List):
        pool = multiprocessing.Pool(len(chunks))
        results = []

        def create_worker_thread(self, chunk):
            with self.acquire_core_thread_affinity():
                # Call the function
                try:
                    return chunk()
                except Exception as e:
                    return e

        try:
            # Set the core affinity every two threads
            threads_spawned = 0
            for chunk in chunks:
                # Schedule the job to the wave process worker
                results.append(
                    pool.apply_async(create_worker_thread, args=(self, chunk))
                )
                threads_spawned += 1
        finally:
            self.pool.close()
            self.pool.join()
            results = [result.get() for result in results]
            # Check if the results are exceptions
            for result in results:
                if isinstance(result, Exception):
                    results = [result]
                    break
        return results

    def run(self):
        print("Wave process worker started")
        while True:
            if WaveProcessWorker.input_queue.empty():
                continue
            print("Wave process worker processing...")
            input = WaveProcessWorker.input_queue.get()
            try:
                output = self.process(input, max_reduce=10)
            except Exception as e:
                output = e
            print("Wave process worker processed")
            WaveProcessWorker.output_queue.put(output)

    def process(self, input, max_reduce=10):
        if max_reduce <= 0:
            raise Exception("Max reduce exceeded")
        # Process the operation on the wave process worker
        # Chunk the data by calling the function's chunk method
        args = input.args
        chunks = input.func.chunk(args, WaveProcessWorker.num_threads_free.value)
        # Perform the operation on the chunks
        results = input.func.reduce(self.wave_process(chunks), input.func.output)
        # Reduce the results by calling the function's reduce method
        # This will return another wave process job
        if len(results) > 1:
            # Process the output again, recursively. Reduce the results until there is only one result.
            output = self.process(output, max_reduce=max_reduce - 1)
        elif len(results) == 1:
            output = results[0]
        else:
            output = WaveProcessOutputNone()
        return output

    def wave_process(self, chunks: List):
        return self.create_pool(chunks)


class WaveProcess:
    def __init__(self):
        # Create the wave process worker
        self.worker = WaveProcessWorker()

    def start(self, debug=False):
        # Start the wave process worker
        if debug:
            print("Starting wave process worker...")
        self.worker.start()

    def terminate(self, debug=False):
        # Terminate the wave process worker
        if debug:
            print("Terminating wave process worker...")
        self.worker.terminate()

    def schedule(self, func, *args):
        # Schedule the job to the wave process worker
        self.worker.enqueue(WaveProcessJob(func, args))

    def get_result(self, timeout=10):
        # Get the result from the wave process worker
        return self.worker.dequeue(timeout=timeout)

    def __del__(self):
        # Terminate the wave process worker
        self.worker.terminate()


class WaveRunner:
    wave_process: ClassVar[WaveProcess] = WaveProcess()
    active_runners: ClassVar[multiprocessing.Value] = multiprocessing.Value("i", 0)

    def __init__(self, debug=False):
        self.debug = debug

    def __enter__(self):
        # Start the wave process
        if WaveRunner.active_runners.value == 0:
            WaveRunner.wave_process.start(debug=self.debug)
        WaveRunner.active_runners.value += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        WaveRunner.active_runners.value -= 1
        if WaveRunner.active_runners.value == 0:
            # Terminate the wave process
            WaveRunner.wave_process.terminate(debug=self.debug)


class Function:
    def __init__(self, inputs: List[Tensor], output: Tensor):
        self.inputs = inputs
        self.output = output

    def process(self, *args):
        # Schedule the function to the wave process
        WaveRunner.wave_process.schedule(self, *args)
        # Get the result from the wave process
        result = WaveRunner.wave_process.get_result(timeout=2)
        # Check if the result is exception
        if isinstance(result, Exception):
            raise result
        if isinstance(result, WaveProcessOutputNone):
            result = None
        return result

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def __call__(self):
        self.forward()
        self.backward()
