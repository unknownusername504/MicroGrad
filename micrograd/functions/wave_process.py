import time
from typing import ClassVar
import multiprocessing
import os
import psutil
from typing import List

from micrograd.tensors.tensor import Tensor
from micrograd.utils.debug_utils import debug_print


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


class AcquireCoreThreadAffinity:
    # Leave one core free for the main process
    num_cores: ClassVar[int] = multiprocessing.cpu_count() - 1
    num_hyperthreads: ClassVar[int] = 2
    max_threads: ClassVar[int] = num_cores * num_hyperthreads
    # Allow for hyperthreading
    num_threads_free: ClassVar[multiprocessing.Value] = multiprocessing.Value(
        "i", max_threads
    )
    num_threads_unreserved: ClassVar[multiprocessing.Value] = multiprocessing.Value(
        "i", max_threads
    )

    core_thread_acquire_affinity_lock: ClassVar[
        multiprocessing.Lock
    ] = multiprocessing.Lock()
    # Two locks for hyperthreading
    core_thread_affinity_locks: ClassVar[List[multiprocessing.Lock]] = [
        multiprocessing.Lock() for _ in range(max_threads)
    ]

    def __init__(self):
        self.core_thread_id = None

    def __enter__(self):
        self.core_thread_id = self.lock_core_thread_affinity()
        debug_print("enter core_thread_id:", self.core_thread_id)

    def __exit__(self, exc_type, exc_value, traceback):
        debug_print("exit core_thread_id:", self.core_thread_id)
        # Sleep for long enough to cause backup
        time.sleep(1)
        self.unlock_core_thread_affinity(self.core_thread_id)

    def lock_core_thread_affinity(self):
        debug_print("Locking core thread affinity...", flush=True)
        with self.core_thread_acquire_affinity_lock:
            # Find a core that is free
            for core_id in range(self.num_cores):
                for hyperthread_id in range(self.num_hyperthreads):
                    # Calculate the thread id
                    core_thread_id = (core_id * self.num_hyperthreads) + hyperthread_id
                    # Check the lock status
                    if self.core_thread_affinity_locks[core_thread_id].acquire(
                        block=False
                    ):
                        # Return the core id
                        with self.num_threads_free.get_lock():
                            self.num_threads_free.value -= 1
                        # Get the current process
                        proc_id = os.getpid()
                        # Set the core affinity
                        psutil.Process(proc_id).cpu_affinity([core_id])
                        # Below is for Linux
                        # os.sched_setaffinity(0, core_id)
                        debug_print("Locked core thread affinity...", flush=True)
                        return core_thread_id
            # If no cores are free, raise an exception
            raise Exception("No cores are free")

    def unlock_core_thread_affinity(self, core_thread_id):
        debug_print("Unlocking core thread affinity...", flush=True)
        with self.core_thread_acquire_affinity_lock:
            # Unlock the core
            with self.num_threads_free.get_lock():
                self.num_threads_free.value += 1
            self.core_thread_affinity_locks[core_thread_id].release()
        debug_print("Unlocked core thread affinity...", flush=True)


class WaveProcessWorker(multiprocessing.Process):
    # Create the input and output queues
    input_queue: ClassVar[multiprocessing.Queue] = multiprocessing.Queue()
    output_queue: ClassVar[multiprocessing.Queue] = multiprocessing.Queue()

    def __init__(self):
        super().__init__()
        self.num_threads_unreserved = AcquireCoreThreadAffinity.num_threads_unreserved
        self.input_queue = WaveProcessWorker.input_queue
        self.output_queue = WaveProcessWorker.output_queue

    def enqueue(self, input):
        debug_print("Wave process worker enqueuing...")
        self.input_queue.put(input)
        debug_print("Wave process worker enqueued")

    def dequeue(self, timeout=10):
        debug_print("Wave process worker dequeuing...")
        result = self.output_queue.get(timeout=timeout)
        debug_print("Wave process worker dequeued")
        return result

    @staticmethod
    def worker_thread(chunk):
        try:
            core_thread_affinity = AcquireCoreThreadAffinity()
            debug_print("Calling function...", flush=True)
            with core_thread_affinity:
                # Call the function
                result = chunk()
            debug_print("Called function...", flush=True)
            debug_print("result:", result)
        except Exception as e:
            result = e
        return result

    @staticmethod
    def test_result(result):
        # Simple test function that will add 1 to the result
        # debug_print("Testing result...", flush=True)
        return result + 1

    def create_pool(self, chunks: List):
        debug_print("Creating pool...")
        pool = multiprocessing.Pool(len(chunks))
        processes = []

        try:
            # Set the core affinity every two threads
            threads_spawned = 0
            debug_print("Spawning threads...")
            for chunk in chunks:
                debug_print("chunk:", chunk)
                # Schedule the job to the wave process worker
                test_only = False
                if test_only:
                    processes.append(
                        pool.apply_async(
                            WaveProcessWorker.test_result,
                            args=(1,),
                        )
                    )
                else:
                    processes.append(
                        pool.apply_async(
                            WaveProcessWorker.worker_thread,
                            args=(chunk,),
                        )
                    )
                threads_spawned += 1
                debug_print("threads_spawned:", threads_spawned)
        finally:
            debug_print("Closing pool...")
            pool.close()
            debug_print("Joining pool...")
            pool.join()
            debug_print("Collecting pool...")
            results = []
            debug_print("type(processes):", type(processes))
            debug_print("processes:", processes)
            # Check if the results are exceptions
            for process in processes:
                debug_print("Trying to get process result...")
                result = process.get()
                if isinstance(result, Exception):
                    debug_print("Exception:", result)
                    results = [result]
                    break
                debug_print("process result:", result)
                results.append(result)
        debug_print("Pool results:", results)
        return results

    def run(self):
        debug_print("Wave process worker started")
        while True:
            if self.input_queue.empty():
                continue
            debug_print("Wave process worker processing...")
            input = self.input_queue.get()
            output = self.process(input, max_reduce=10)
            debug_print("Wave process worker processed")
            self.output_queue.put(output)

    def process(self, input, max_reduce=10):
        try:
            if max_reduce <= 0:
                raise Exception("Max reduce exceeded")
            # Process the operation on the wave process worker
            # Chunk the data by calling the function's chunk method
            args = input.args
            debug_print("args:", args)
            with self.num_threads_unreserved.get_lock():
                num_threads_unreserved = self.num_threads_unreserved.value
                # Take the free threads
                self.num_threads_unreserved.value = 0
            debug_print("num_threads_unreserved:", num_threads_unreserved)
            chunks = input.func.chunk(args, num_threads_unreserved)
            debug_print("chunks:", chunks)
            # Add back unused threads
            with self.num_threads_unreserved.get_lock():
                self.num_threads_unreserved.value += num_threads_unreserved - len(
                    chunks
                )
            # Perform the operation on the chunks
            debug_print("Performing reduce on chunks...")
            results = input.func.reduce(self.wave_process(chunks), input.func.output)
            debug_print("results:", results)
            # Reduce the results by calling the function's reduce method
            # This will return another wave process job
            if len(results) > 1:
                debug_print("Results require reduce")
                # Process the output again, recursively. Reduce the results until there is only one result.
                output = self.process(output, max_reduce=max_reduce - 1)
            elif len(results) == 1:
                output = results[0]
            else:
                output = WaveProcessOutputNone()
        except Exception as e:
            output = e
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
            debug_print("Starting wave process worker...")
        self.worker.start()
        # Wait for the wave process worker to start
        max_wait = 10
        while not self.worker.is_alive():
            time.sleep(1)
            debug_print("Waiting for wave process worker to start...")
            max_wait -= 1
        if max_wait <= 0:
            raise Exception("Max wait exceeded")

    def terminate(self, debug=False):
        # Terminate the wave process worker
        if debug:
            debug_print("Terminating wave process worker...")
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
        self.wave_process = WaveRunner.wave_process
        self.active_runners = WaveRunner.active_runners

    def __enter__(self):
        # Start the wave process
        with self.active_runners.get_lock():
            if self.active_runners.value == 0:
                self.wave_process.start(debug=self.debug)
            self.active_runners.value += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        with self.active_runners.get_lock():
            self.active_runners.value -= 1
            if self.active_runners.value == 0:
                # Terminate the wave process
                self.wave_process.terminate(debug=self.debug)

    @staticmethod
    def process(*args):
        # Check if the wave process is running
        with WaveRunner.active_runners.get_lock():
            if WaveRunner.active_runners.value == 0:
                return Exception("Wave process is not running")
        # Schedule the function to the wave process
        WaveRunner.wave_process.schedule(*args)
        # Get the result from the wave process
        return WaveRunner.wave_process.get_result(timeout=2)


class Function:
    def __init__(self, inputs: List[Tensor], output: Tensor):
        self.inputs = inputs
        self.output = output

    def process(self, *args):
        result = WaveRunner.process(self, *args)
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
        result = self.forward()
        self.backward()
        return result
