"""Application lifecycle management."""
from contextlib import asynccontextmanager
import math
import os
from pathlib import Path
from typing import Iterable, List, Optional

import torch
import yaml

from core.logger import get_logger
from services.pipeline_orchestrator import PipelineOrchestrator
from workers.worker_manager import WorkerManager

logger = get_logger("lifecycle")
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "production.yaml"


class AppState:
    def __init__(self):
        self.worker_manager = None
        self.pipeline = None


app_state = AppState()


def detect_devices() -> List[str]:
    """Detect available devices."""
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        devices = [f"cuda:{i}" for i in range(count)]
        logger.info(f"Detected {count} GPUs: {devices}")
        return devices

    logger.warning("No GPU detected. Using CPU mode.")
    return ["cpu"]


def _load_runtime_config() -> dict:
    config_path = os.getenv("ASR_CONFIG_PATH")
    path = Path(config_path).expanduser() if config_path else DEFAULT_CONFIG_PATH

    if not path.exists():
        logger.warning(f"Config file not found: {path}. Using defaults.")
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            logger.warning(f"Config content is not a mapping: {path}. Using defaults.")
            return {}
        logger.info(f"Loaded runtime config from {path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load config from {path}: {str(e)}", exc_info=True)
        return {}


def _normalize_model_names(model_names: Iterable[str]) -> List[str]:
    names = [name.strip() for name in model_names if isinstance(name, str) and name.strip()]
    if not names:
        raise ValueError("model_names must contain at least one valid model name")
    return names


def _normalize_devices(devices: Optional[List[str]]) -> List[str]:
    if devices is None:
        raw_devices = []
    elif isinstance(devices, str):
        raw_devices = [devices]
    else:
        raw_devices = list(devices)

    normalized = [device.strip() for device in raw_devices if isinstance(device, str) and device.strip()]
    normalized = list(dict.fromkeys(normalized))
    if not normalized:
        raise ValueError("devices must contain at least one valid device")
    return normalized


def _align_devices_with_available(requested_devices: List[str], available_devices: List[str]) -> List[str]:
    if not available_devices:
        raise ValueError("No available devices detected")

    available_unique = list(dict.fromkeys(available_devices))
    selected = []
    invalid_devices = []

    for device in requested_devices:
        if device in available_unique and device not in selected:
            selected.append(device)
        elif device not in available_unique:
            invalid_devices.append(device)

    if not invalid_devices and selected:
        return selected

    if invalid_devices:
        logger.warning(
            "Requested devices are not available and will be replaced: "
            + f"{invalid_devices}. Available devices: {available_unique}"
        )

    remaining_available = [device for device in available_unique if device not in selected]
    needed = max(0, len(requested_devices) - len(selected))

    if len(remaining_available) < needed:
        logger.warning(
            "Requested device count exceeds available unique devices. "
            f"requested={requested_devices}, available={available_unique}. "
            "Falling back to all available devices."
        )
        return available_unique

    selected.extend(remaining_available[:needed])
    logger.warning(f"Fallback devices selected: {selected}")
    return selected


def _resolve_model_names(runtime_config: dict, model_names: Optional[Iterable[str]]) -> List[str]:
    if model_names is not None:
        return _normalize_model_names(model_names)

    models_cfg = runtime_config.get("models", {})
    if not isinstance(models_cfg, dict):
        models_cfg = {}

    config_model_names = models_cfg.get(
        "load",
        [
            "paraformer_qf",
            "whisper_turbo",
            "paraformer_zh",
            "sensevoice",
            "wenet_speech",
            "funasr_nano",
        ],
    )
    return _normalize_model_names(config_model_names)


def _resolve_devices(runtime_config: dict, devices: Optional[List[str]]) -> List[str]:
    available_devices = detect_devices()

    if devices is not None:
        requested_devices = _normalize_devices(devices)
        return _align_devices_with_available(requested_devices, available_devices)

    gpu_cfg = runtime_config.get("gpu", {})
    if not isinstance(gpu_cfg, dict):
        gpu_cfg = {}

    auto_detect = gpu_cfg.get("auto_detect", True)
    if isinstance(auto_detect, str):
        auto_detect = auto_detect.strip().lower() in {"1", "true", "yes", "on"}

    if auto_detect:
        return available_devices

    configured_devices = gpu_cfg.get("devices", [])
    requested_devices = _normalize_devices(configured_devices)
    return _align_devices_with_available(requested_devices, available_devices)


def _resolve_max_models_per_worker(runtime_config: dict) -> int:
    models_cfg = runtime_config.get("models", {})
    if not isinstance(models_cfg, dict):
        models_cfg = {}

    value = models_cfg.get("max_models_per_worker", 2)
    try:
        parsed = int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"models.max_models_per_worker must be an integer, got: {value}") from e

    if parsed <= 0:
        raise ValueError("models.max_models_per_worker must be > 0")
    return parsed


def _fit_models_to_devices(model_names: List[str], devices: List[str], max_models_per_worker: int) -> int:
    if not devices:
        raise ValueError("devices must not be empty")

    required = max(1, math.ceil(len(model_names) / len(devices)))
    if max_models_per_worker >= required:
        return max_models_per_worker

    logger.warning(
        "Configured max_models_per_worker is insufficient for current model/device count. "
        f"Auto-adjusting from {max_models_per_worker} to {required} "
        f"(models={len(model_names)}, devices={len(devices)})."
    )
    return required


def initialize_system(model_names: Optional[Iterable[str]] = None, devices: Optional[List[str]] = None):
    if app_state.worker_manager is not None and app_state.pipeline is not None:
        logger.info("System already initialized, skipping re-initialization")
        return

    if app_state.worker_manager is not None or app_state.pipeline is not None:
        logger.warning("Detected partial initialization state, resetting before re-initialization")
        if app_state.worker_manager is not None:
            try:
                app_state.worker_manager.shutdown()
            except Exception as e:
                logger.error(f"Failed to cleanup previous workers: {str(e)}", exc_info=True)
        app_state.worker_manager = None
        app_state.pipeline = None

    logger.info("====== System Initialization Started ======")

    runtime_config = _load_runtime_config()
    normalized_model_names = _resolve_model_names(runtime_config, model_names)
    final_devices = _resolve_devices(runtime_config, devices)
    max_models_per_worker = _resolve_max_models_per_worker(runtime_config)
    max_models_per_worker = _fit_models_to_devices(
        normalized_model_names,
        final_devices,
        max_models_per_worker,
    )

    logger.info(f"Using models: {normalized_model_names}")
    logger.info(f"Using devices: {final_devices}")
    logger.info(f"Using max_models_per_worker: {max_models_per_worker}")

    worker_manager = None

    try:
        logger.info("Starting worker processes...")
        worker_manager = WorkerManager(
            model_names=normalized_model_names,
            devices=final_devices,
            max_models_per_worker=max_models_per_worker,
        )
        logger.info(f"Worker processes started: {len(worker_manager.processes)} processes")

        logger.info("Initializing pipeline orchestrator...")
        pipeline = PipelineOrchestrator(worker_manager, runtime_config=runtime_config)

        app_state.worker_manager = worker_manager
        app_state.pipeline = pipeline
    except Exception:
        if worker_manager is not None:
            try:
                worker_manager.shutdown()
            except Exception as e:
                logger.error(f"Failed to rollback workers after init error: {str(e)}", exc_info=True)
        app_state.worker_manager = None
        app_state.pipeline = None
        raise

    logger.info("====== System Initialization Complete ======")


@asynccontextmanager
async def lifespan(app):
    try:
        initialize_system()
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Failed to initialize system: {str(e)}", exc_info=True)
        raise

    yield

    logger.info("====== Application Shutdown Started ======")
    try:
        if app_state.worker_manager:
            logger.info("Shutting down worker processes...")
            app_state.worker_manager.shutdown()
            logger.info("Worker processes shutdown complete")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU memory cleared")
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}", exc_info=True)
    finally:
        app_state.worker_manager = None
        app_state.pipeline = None
        logger.info("====== Application Shutdown Complete ======")
