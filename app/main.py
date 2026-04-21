"""
FastAPI 主应用 - 优化版本
异步I/O + 内存优化
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
import tempfile
import os
import aiofiles
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

from app.lifecycle import lifespan, app_state
from infra.concurrency import RequestLimiter
from core.logger import get_logger
from core.resource_monitor import ResourceMonitor

logger = get_logger("main")

# 初始化应用
app = FastAPI(
    lifespan=lifespan,
    title="Enterprise ASR API",
    version="1.0.0"
)

# 并发限流器
limiter = RequestLimiter(max_concurrent=16)

# 资源监控
monitor = ResourceMonitor()

# 创建线程池用于同步操作
executor = ThreadPoolExecutor(max_workers=4)

# 常量配置
SUPPORTED_FORMATS = ('.wav', '.mp3', '.m4a', '.flac')
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
CHUNK_SIZE = 1024 * 64  # 64KB


def _safe_queue_size(queue_obj):
    try:
        return queue_obj.qsize()
    except (AttributeError, NotImplementedError, OSError):
        return None


@app.post("/asr")
async def asr_endpoint(file: UploadFile = File(...)):
    """
    语音识别 API 端点 - 优化版
    
    优化点：
    1. 异步文件I/O
    2. 文件大小检查
    3. 资源监控日志
    4. 更详细的错误信息
    
    Args:
        file: 上传的音频文件
    
    Returns:
        JSON 格式的识别结果
    """
    filename = (file.filename or "").strip()
    if not filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    request_id = f"{datetime.now().isoformat()}_{filename}"
    logger.info(f"[{request_id}] Received ASR request")
    
    tmp_path = None
    
    try:
        # 1. 验证文件类型
        if not filename.lower().endswith(SUPPORTED_FORMATS):
            logger.warning(
                f"[{request_id}] Invalid format: {filename}"
            )
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format. Supported: {SUPPORTED_FORMATS}"
            )
        
        # 2. 异步读取文件到临时路径
        logger.info(f"[{request_id}] Reading file asynchronously")
        
        with tempfile.NamedTemporaryFile(
            suffix=os.path.splitext(filename)[1],
            delete=False
        ) as tmp:
            tmp_path = tmp.name
        
        # 异步写入文件（避免阻塞）
        file_size = 0
        async with aiofiles.open(tmp_path, 'wb') as f:
            while chunk := await file.read(CHUNK_SIZE):
                file_size += len(chunk)
                
                # 检查文件大小
                if file_size > MAX_FILE_SIZE:
                    logger.warning(
                        f"[{request_id}] File too large: {file_size} bytes"
                    )
                    raise HTTPException(
                        status_code=413,
                        detail=f"File size exceeds {MAX_FILE_SIZE} bytes"
                    )
                
                await f.write(chunk)
        
        logger.info(
            f"[{request_id}] File saved: {file_size} bytes"
        )
        
        # 3. 记录资源使用前状态
        gpu_memory_before = monitor.gpu_memory()
        
        # 4. Submit to pipeline in thread pool
        logger.info(f"[{request_id}] Submitting to pipeline")

        if app_state.pipeline is None:
            raise HTTPException(
                status_code=503,
                detail="ASR pipeline is not initialized"
            )

        loop = asyncio.get_running_loop()
        source_ref = f"upload://{request_id}/{filename}"
        result = await limiter.run(
            loop.run_in_executor(
                executor,
                app_state.pipeline.run_file,
                tmp_path,
                source_ref,
            )
        )
        
        # 5. 记录资源使用后状态
        gpu_memory_after = monitor.gpu_memory()
        logger.info(
            f"[{request_id}] GPU memory: "
            f"{gpu_memory_before} -> {gpu_memory_after}"
        )
        
        logger.info(f"[{request_id}] ASR completed successfully")
        
        return {
            "success": True,
            "request_id": request_id,
            "data": result
        }
    
    except HTTPException:
        raise
    
    except TimeoutError as e:
        logger.error(f"[{request_id}] Timeout: {str(e)}")
        raise HTTPException(
            status_code=504,
            detail="ASR processing timeout"
        )
    
    except Exception as e:
        logger.error(
            f"[{request_id}] Error: {str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {str(e)}"
        )
    
    finally:
        # 清理临时文件
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                logger.debug(f"[{request_id}] Temp file cleaned")
            except Exception as e:
                logger.warning(f"[{request_id}] Failed to clean temp: {str(e)}")


@app.get("/health")
async def health_check():
    """
    健康检查端点
    
    返回：
    - 系统状态
    - GPU内存使用情况
    - Worker进程状态
    """
    worker_manager = app_state.worker_manager
    if worker_manager is None:
        return {
            "status": "initializing",
            "timestamp": datetime.now().isoformat(),
            "workers": {"total": 0, "alive": 0},
            "gpu_memory": monitor.gpu_memory(),
            "pipeline": app_state.pipeline is not None
        }

    total_workers = len(worker_manager.processes)
    alive_workers = sum(1 for p in worker_manager.processes if p.is_alive())
    status = "healthy" if total_workers > 0 and alive_workers > 0 else "degraded"

    return {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "workers": {
            "total": total_workers,
            "alive": alive_workers
        },
        "gpu_memory": monitor.gpu_memory(),
        "pipeline": app_state.pipeline is not None
    }


@app.get("/metrics")
async def metrics():
    """
    资源监控端点
    返回当前系统资源使用情况
    """
    worker_manager = app_state.worker_manager
    task_queue_size = None
    task_queue_sizes = None
    if worker_manager is not None:
        if hasattr(worker_manager, "total_task_queue_size"):
            task_queue_size = worker_manager.total_task_queue_size()
        else:
            task_queue_size = _safe_queue_size(worker_manager.task_queue)
        if hasattr(worker_manager, "task_queue_sizes"):
            task_queue_sizes = worker_manager.task_queue_sizes()

    return {
        "timestamp": datetime.now().isoformat(),
        "gpu_memory": monitor.gpu_memory(),
        "task_queue_size": task_queue_size,
        "task_queue_sizes": task_queue_sizes,
        "result_queue_size": None if worker_manager is None else _safe_queue_size(worker_manager.result_queue)
    }

