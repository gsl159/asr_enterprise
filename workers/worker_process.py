"""Worker 子进程入口。

职责：
1. 根据分配到的 device 初始化运行环境（GPU/CPU）。
2. 在子进程内加载模型，避免主进程持有大模型占用显存。
3. 循环消费任务队列中的音频任务，执行多模型并行推理。
4. 将推理结果写回结果队列，供主进程聚合。
5. 收到 STOP 信号后优雅退出并释放资源。
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Queue
from queue import Full

import torch

from core.gpu_manager import GPUManager
from core.logger import get_logger
from models.factory import ModelFactory

logger = get_logger("worker_process")


def worker_loop(device: str, model_names, task_queue: Queue, result_queue: Queue):
    """Worker 主循环函数（运行在独立子进程中）。

    Args:
        device: 目标设备标识，例如 "cuda:0" 或 "cpu"。
        model_names: 需要加载的模型名称列表。
        task_queue: 任务队列，元素格式通常为 (task_id, audio_path)。
        result_queue: 结果队列，元素格式为 (task_id, model_results)。
    """
    executor = None

    try:
        # 设备初始化：
        # 对于 cuda:N，收窄可见设备到单卡后，子进程内统一使用 cuda:0 访问。
        # 这样可减少多卡索引混乱，并确保每个 worker 绑定到指定物理卡。
        if device.startswith("cuda:"):
            gpu_id = device.split(":")[-1]
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
            # 可见卡被限制为 1 张后，当前进程内的逻辑设备索引始终为 cuda:0。
            runtime_device = "cuda:0"
            logger.info(
                f"Worker started on {device}, GPU_ID={gpu_id}, runtime_device={runtime_device}"
            )
        else:
            # CPU 模式下显式移除 CUDA 可见性配置，避免继承脏环境变量。
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            runtime_device = "cpu"
            logger.info("Worker started on CPU")

        # 每个 worker 仅持有本地 runtime_device，实现进程级设备隔离。
        gpu_manager = GPUManager(auto_detect=False, devices=[runtime_device])

        # 在子进程内加载模型，保证显存由子进程独占和管理。
        logger.info(f"Loading models: {model_names}")
        models = ModelFactory.create(model_names, gpu_manager)
        if not models:
            raise RuntimeError("No models were created for this worker")
        logger.info(f"Models loaded on {device}")

        # 子进程内使用线程池并行执行“多模型同任务”推理。
        # 线程数至少为 1，且默认等于当前 worker 成功加载的模型数量。
        executor = ThreadPoolExecutor(max_workers=max(1, len(models)))

        while True:
            # 阻塞读取任务；队列由 WorkerManager 负责投递。
            task = task_queue.get()
            if task == "STOP":
                logger.info("Received STOP signal")
                break

            # 防御式校验任务数据结构，避免脏数据导致进程崩溃。
            if not isinstance(task, tuple) or len(task) != 2:
                logger.warning(f"Skipping invalid task payload: {task!r}")
                continue

            task_id, audio_path = task
            logger.info(f"Task {task_id}: Processing {audio_path}")
            start_time = time.time()

            try:
                # 单模型推理包装：保证每个模型的异常被局部捕获，不影响其他模型执行。
                def infer_model(model):
                    try:
                        text = model.infer(audio_path)
                        return {"model": model.model_name, "text": text, "error": None}
                    except Exception as e:
                        logger.error(f"Task {task_id}: {model.model_name} failed: {str(e)}")
                        return {"model": model.model_name, "text": None, "error": str(e)}

                # 一个任务提交给多个模型并行推理。
                futures = [executor.submit(infer_model, model) for model in models]

                results = []
                for future in futures:
                    try:
                        # 给每个 future 设置超时，防止单模型卡死拖垮整任务。
                        results.append(future.result(timeout=300))
                    except Exception as e:
                        logger.error(f"Task {task_id}: Future result failed: {str(e)}")
                        results.append({"model": "unknown", "text": None, "error": str(e)})

                # 将完整模型结果写回主进程，供 pipeline 聚合。
                result_queue.put((task_id, results), timeout=5.0)
                elapsed = time.time() - start_time
                logger.info(f"Task {task_id} completed in {elapsed:.2f}s")
            except Exception as e:
                # 任务级别失败也尽量写回 error 结果，避免主进程一直等待超时。
                logger.error(f"Task {task_id} failed: {str(e)}", exc_info=True)
                try:
                    result_queue.put((task_id, {"error": str(e)}), timeout=5.0)
                except Full as full_e:
                    # 若错误结果都无法写回，说明结果通道堵塞，升级为致命错误。
                    logger.critical(f"Task {task_id}: result queue full while reporting error")
                    raise RuntimeError("Result queue is full") from full_e
    except Full as e:
        # 结果队列满属于系统级压力问题，交由上层拉闸/重启策略处理。
        logger.error(f"Worker {device} failed to publish result due to full queue: {str(e)}", exc_info=True)
        raise
    except Exception as e:
        # 任何未处理异常都视为 worker 崩溃，抛出供父进程感知。
        logger.error(f"Worker crashed on {device}: {str(e)}", exc_info=True)
        raise
    finally:
        # 退出前先回收线程池。
        if executor:
            executor.shutdown(wait=True)
            logger.info("Thread pool shutdown complete")

        # 清理 CUDA 缓存，减少残留显存。
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 子进程生命周期结束日志。
        logger.info(f"Worker on {device} shutdown complete")
