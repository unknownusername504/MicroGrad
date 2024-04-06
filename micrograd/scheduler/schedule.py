import time
from typing import ClassVar, Optional
import multiprocessing
import os
import psutil
from typing import List

from micrograd.tensors.tensor import Function
from micrograd.utils.debug_utils import debug_print


# Class representing a wave process output of none
class WaveProcessOutputNone:
    def __init__(self):
        pass


# Class with wave process structure
class WaveProcessJob:
    # Class with wave process job
    def __init__(self, func):
        self.func = func


class CoreThreadAffinityManager:
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

    core_thread_acquire_affinity_lock: ClassVar[multiprocessing.Lock] = (
        multiprocessing.Lock()
    )
    # Two locks for hyperthreading
    core_thread_affinity_locks: ClassVar[List[multiprocessing.Lock]] = [
        multiprocessing.Lock() for _ in range(max_threads)
    ]

    def __init__(self):
        self.core_thread_id = None

    def __enter__(self, core_thread_id: Optional[int] = None):
        if core_thread_id is not None:
            self.core_thread_id = core_thread_id
        else:
            self.core_thread_id = self.lock_core_thread_affinity()
        CoreThreadAffinityManager.set_thread_affinity(self.core_thread_id)
        debug_print("enter core_thread_id:", self.core_thread_id)

    def __exit__(self, exc_type, exc_value, traceback):
        debug_print("exit core_thread_id:", self.core_thread_id)
        # Sleep for long enough to cause backup
        # time.sleep(1)
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

    def get_core_thread_id(self):
        return self.core_thread_id

    @staticmethod
    def set_thread_affinity(core_thread_id):
        # Get the current process
        proc_id = os.getpid()
        # Set the core affinity
        psutil.Process(proc_id).cpu_affinity([core_thread_id])


class WaveProcessWorker(multiprocessing.Process):
    # Create the input and output queues
    input_queue: ClassVar[multiprocessing.Queue] = multiprocessing.Queue()
    output_queue: ClassVar[multiprocessing.Queue] = multiprocessing.Queue()

    def __init__(
        self, wave_runner, core_thread_affinity_manager: CoreThreadAffinityManager
    ):
        super().__init__()
        self.wave_runner = wave_runner
        self.core_thread_affinity_manager = core_thread_affinity_manager
        self.num_threads_unreserved: multiprocessing.Value = (
            core_thread_affinity_manager.num_threads_unreserved
        )
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
    def worker_thread(
        func: Function, core_thread_id, result_queue: multiprocessing.Queue
    ):
        try:
            CoreThreadAffinityManager.set_thread_affinity(core_thread_id)
            debug_print("\nCalling function...", flush=True)
            debug_print("func x:", func.inputs[0], flush=True)
            debug_print("func y:", func.inputs[1], flush=True)
            # Call the function
            result = func()
            debug_print("func result:", result, flush=True)
            debug_print("Called function...\n", flush=True)
        except Exception as e:
            result = e
        result_queue.put(result)

    def create_process(self, func: Function):
        debug_print("Creating process...")

        try:
            # Set the core affinity every two threads
            debug_print("Spawning threads...")
            locked_core_thread_id = []
            debug_print("func:", func)
            # Schedule the job to the wave process worker
            locked_core_thread_id = (
                self.core_thread_affinity_manager.lock_core_thread_affinity()
            )
            result_queue = multiprocessing.Queue()
            process = multiprocessing.Process(
                target=WaveProcessWorker.worker_thread,
                args=(func, locked_core_thread_id, result_queue),
            )
            process.start()
        finally:
            debug_print("Joining process...")
            process.join()
            debug_print("Closing process...")
            process.close()
            self.core_thread_affinity_manager.unlock_core_thread_affinity(
                locked_core_thread_id
            )
            debug_print("Collecting process...")
            # Check if the results are exceptions
            debug_print("Trying to get process result...")
            result = result_queue.get()
            if isinstance(result, Exception):
                debug_print("Exception:", result)
            debug_print("process result:", result)
        return result

    def run(self):
        debug_print("Wave process worker started")
        while True:
            if self.input_queue.empty():
                continue
            debug_print("Wave process worker processing...")
            input = self.input_queue.get()
            output = self.process(input)
            debug_print("Wave process worker output:", output)
            debug_print("Wave process worker processed")
            self.output_queue.put(output)

    def process(self, input):
        try:
            # Process the operation on the wave process worker
            debug_print("func:", input.func)
            # Reserve a thread
            with self.num_threads_unreserved.get_lock():
                self.num_threads_unreserved.value -= 1
            output = self.wave_process(input.func)
            # Check if the output is exception
            if isinstance(output, Exception):
                raise output
            input.func.output = output
        except Exception as e:
            debug_print("Exception occurred during wave process:", e)
            output = e
        debug_print("output:", output)
        return output

    def wave_process(self, chunks: List):
        return self.create_process(chunks)


class WaveProcess:
    def __init__(self):
        self.worker = None

    def start(self, wave_runner, core_thread_affinity_manager):
        # Create the wave process worker
        self.worker = WaveProcessWorker(wave_runner, core_thread_affinity_manager)
        # Start the wave process worker
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

    def terminate(self):
        # Terminate the wave process worker
        debug_print("Terminating wave process worker...")
        self.worker.terminate()

    def schedule(self, func):
        # Schedule the job to the wave process worker
        self.worker.enqueue(WaveProcessJob(func))

    def get_result(self, timeout=10):
        # Get the result from the wave process worker
        return self.worker.dequeue(timeout=timeout)

    def __del__(self):
        # Terminate the wave process worker
        self.worker.terminate()
        self.worker.join()
        self.worker.close()
        self.worker = None


class WaveRunner:
    wave_process: ClassVar[WaveProcess] = WaveProcess()
    active_runners: ClassVar[multiprocessing.Value] = multiprocessing.Value("i", 0)
    core_thread_affinity_manager: ClassVar[CoreThreadAffinityManager] = (
        CoreThreadAffinityManager()
    )

    def __init__(self):
        self.wave_process = WaveRunner.wave_process
        self.active_runners = WaveRunner.active_runners

    def __enter__(self):
        # Start the wave process
        with self.active_runners.get_lock():
            if self.active_runners.value == 0:
                debug_print("Starting wave process...")
                self.wave_process.start(self, WaveRunner.core_thread_affinity_manager)
            self.active_runners.value += 1
            debug_print("Active runners:", self.active_runners.value)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        with self.active_runners.get_lock():
            self.active_runners.value -= 1
            debug_print("Active runners:", self.active_runners.value)
            if self.active_runners.value == 0:
                debug_print("Terminating wave process...")
                # Terminate the wave process
                self.wave_process.terminate()

    def send_function(self, function: Function):
        # Check if the wave process is running
        with self.active_runners.get_lock():
            if self.active_runners.value == 0:
                return Exception("Wave process is not running")
        # Schedule the function to the wave process
        self.wave_process.schedule(function)
        # Get the result from the wave process
        return self.wave_process.get_result(timeout=2)
