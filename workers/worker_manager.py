"""Worker process management."""
import multiprocessing as mp
import threading
import time
from queue import Empty, Full
from typing import List

from core.logger import get_logger
from workers.worker_process import worker_loop

logger = get_logger("worker_manager")


class WorkerManager:
    """Multi-process worker manager.

    Model groups are assigned to devices. Each worker owns one task queue;
    all workers share one result queue.
    """

    def __init__(
        self,
        model_names,
        devices,
        max_models_per_worker: int = 2,
        startup_grace_seconds: float = 8.0,
    ):
        if not model_names:
            raise ValueError("model_names must not be empty")
        if not devices:
            raise ValueError("devices must not be empty")
        if not isinstance(max_models_per_worker, int) or max_models_per_worker <= 0:
            raise ValueError("max_models_per_worker must be a positive integer")

        self.max_models_per_worker = max_models_per_worker
        self.model_groups = self._split_model_groups(model_names, max_models_per_worker)

        if len(self.model_groups) > len(devices):
            raise ValueError(
                "Not enough devices for model groups. "
                f"Need {len(self.model_groups)} devices, got {len(devices)}. "
                f"max_models_per_worker={max_models_per_worker}, model_count={len(model_names)}"
            )

        self.assigned_devices = self._select_assigned_devices(devices, len(self.model_groups))

        self.mp_ctx = mp.get_context("spawn")
        self.task_queues = []
        self.result_queue = self.mp_ctx.Queue(maxsize=1000)
        self.processes = []
        self.worker_specs = []

        self.task_queue = None  # backward compatibility
        self.task_id = 0
        self.next_worker = 0
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self._reported_dead = set()

        logger.info(
            f"Starting workers: groups={len(self.model_groups)}, "
            f"max_models_per_worker={self.max_models_per_worker}, devices={self.assigned_devices}"
        )

        try:
            for worker_id, (device, group_models) in enumerate(zip(self.assigned_devices, self.model_groups)):
                task_queue = self.mp_ctx.Queue(maxsize=1000)
                process = self.mp_ctx.Process(
                    target=worker_loop,
                    args=(device, group_models, task_queue, self.result_queue),
                    name=f"Worker-{worker_id}-{device}",
                )
                process.start()

                self.task_queues.append(task_queue)
                self.processes.append(process)
                self.worker_specs.append(
                    {
                        "worker_id": worker_id,
                        "device": device,
                        "models": group_models,
                        "process": process,
                    }
                )
                logger.info(f"Worker {worker_id} started on {device} with models: {group_models}")

            self._ensure_workers_healthy(startup_grace_seconds=startup_grace_seconds)

            if self.task_queues:
                self.task_queue = self.task_queues[0]

            self.health_check_thread = threading.Thread(target=self._health_check, daemon=True)
            self.health_check_thread.start()
        except Exception:
            self._cleanup_partial_startup()
            raise

        logger.info(f"All {len(self.processes)} workers started")

    @staticmethod
    def _split_model_groups(model_names: List[str], group_size: int) -> List[List[str]]:
        return [model_names[i:i + group_size] for i in range(0, len(model_names), group_size)]

    @staticmethod
    def _select_assigned_devices(devices: List[str], required_count: int) -> List[str]:
        assigned = list(devices[:required_count])
        cuda_devices = [d for d in assigned if isinstance(d, str) and d.startswith("cuda:")]
        if len(set(cuda_devices)) != len(cuda_devices):
            raise ValueError(
                "Duplicate CUDA devices detected in assigned workers. "
                "Each worker must use a unique CUDA device to avoid OOM risk. "
                f"assigned={assigned}"
            )
        return assigned

    def _ensure_workers_healthy(self, startup_grace_seconds: float):
        deadline = time.time() + max(0.5, float(startup_grace_seconds))
        while time.time() < deadline:
            dead = [spec for spec in self.worker_specs if not spec["process"].is_alive()]
            if not dead:
                return
            time.sleep(0.2)

        dead = [spec for spec in self.worker_specs if not spec["process"].is_alive()]
        if dead:
            details = ", ".join(
                f"{spec['process'].name}(device={spec['device']}, models={spec['models']}, "
                f"exitcode={spec['process'].exitcode})"
                for spec in dead
            )
            raise RuntimeError(
                "Some worker processes failed during startup. "
                "Please check model dependencies and model paths. "
                f"dead_workers=[{details}]"
            )

    def _cleanup_partial_startup(self):
        self.stopped.set()
        for queue_obj in self.task_queues:
            try:
                queue_obj.put("STOP", timeout=0.2)
            except Exception:
                pass

        for process in self.processes:
            try:
                process.join(timeout=1)
            except Exception:
                pass
            if process.is_alive():
                try:
                    process.terminate()
                except Exception:
                    pass

    def _health_check(self):
        while not self.stopped.wait(timeout=30):
            for spec in self.worker_specs:
                process = spec["process"]
                if not process.is_alive() and process.name not in self._reported_dead:
                    self._reported_dead.add(process.name)
                    logger.warning(
                        f"Process {process.name} is dead "
                        f"(device={spec['device']}, models={spec['models']}, exitcode={process.exitcode})"
                    )

    def _alive_worker_ids(self) -> List[int]:
        alive = []
        for spec in self.worker_specs:
            if spec["process"].is_alive():
                alive.append(spec["worker_id"])
        return alive

    def _submit_to_worker(self, worker_id: int, task_id: int, audio_path: str):
        if worker_id < 0 or worker_id >= len(self.task_queues):
            raise ValueError(f"Invalid worker_id: {worker_id}")

        try:
            self.task_queues[worker_id].put((task_id, audio_path), timeout=5.0)
        except Full as e:
            raise TimeoutError(f"Task queue of worker {worker_id} is full. Please retry later.") from e

    def submit(self, audio_path: str, worker_id: int = None):
        with self.lock:
            alive_worker_ids = self._alive_worker_ids()
            if not alive_worker_ids:
                raise RuntimeError("No active worker process available")

            if worker_id is None:
                selected_worker = alive_worker_ids[self.next_worker % len(alive_worker_ids)]
                self.next_worker = (self.next_worker + 1) % len(alive_worker_ids)
            else:
                selected_worker = worker_id
                if selected_worker not in alive_worker_ids:
                    raise RuntimeError(f"Worker {selected_worker} is not alive")

            self.task_id += 1
            task_id = self.task_id

        self._submit_to_worker(selected_worker, task_id, audio_path)
        logger.info(f"Task {task_id} submitted to worker {selected_worker}: {audio_path}")
        return task_id

    def submit_to_all(self, audio_path: str) -> List[int]:
        task_ids = []

        with self.lock:
            alive_worker_ids = self._alive_worker_ids()
            if not alive_worker_ids:
                raise RuntimeError("No active worker process available")

            for worker_id in alive_worker_ids:
                self.task_id += 1
                task_id = self.task_id
                task_ids.append((worker_id, task_id))

        submitted_ids = []
        for worker_id, task_id in task_ids:
            self._submit_to_worker(worker_id, task_id, audio_path)
            submitted_ids.append(task_id)

        logger.info(f"Broadcast task submitted to workers: task_ids={submitted_ids}, audio={audio_path}")
        return submitted_ids

    def get_result(self, timeout=None):
        try:
            return self.result_queue.get(timeout=timeout)
        except Empty:
            raise
        except Exception as e:
            logger.error(f"Failed to get result: {str(e)}")
            raise

    def task_queue_sizes(self):
        sizes = []
        for queue_obj in self.task_queues:
            try:
                sizes.append(queue_obj.qsize())
            except (NotImplementedError, OSError, AttributeError):
                sizes.append(None)
        return sizes

    def total_task_queue_size(self):
        sizes = self.task_queue_sizes()
        known_sizes = [size for size in sizes if isinstance(size, int)]
        if not known_sizes:
            return None
        return sum(known_sizes)

    def shutdown(self):
        self.stopped.set()
        logger.info(f"Shutting down {len(self.processes)} workers")

        for queue_obj in self.task_queues:
            try:
                queue_obj.put("STOP", timeout=2.0)
            except Exception as e:
                logger.warning(f"Failed to send STOP: {str(e)}")

        for process in self.processes:
            process.join(timeout=10)
            if process.is_alive():
                logger.warning(f"Force terminating {process.name}")
                process.terminate()

        logger.info("All workers shutdown")

