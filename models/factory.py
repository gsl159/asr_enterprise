"""Model factory for ASR model instances."""
from importlib import import_module
from typing import List

from core.logger import get_logger
from models.registry import MODEL_REGISTRY, get_model_class

logger = get_logger("model_factory")

MODEL_MODULE_MAP = {
    "paraformer_qf": "models.paraformer_qf",
    "whisper_turbo": "models.whisper_turbo",
    "paraformer_zh": "models.paraformer_zh",
    "sensevoice": "models.sensevoice",
    "wenet_speech": "models.wenet_speech",
    "funasr_nano": "models.funasr_nano",
}


class ModelFactory:
    """Create model instances from registered plugins."""

    @staticmethod
    def create(model_names: List[str], gpu_manager):
        models = []
        failures = []

        for name in model_names:
            logger.info(f"Creating model: {name}")

            if name not in MODEL_REGISTRY:
                module_name = MODEL_MODULE_MAP.get(name, f"models.{name}")
                try:
                    import_module(module_name)
                except Exception as e:
                    msg = f"Failed to import model plugin '{name}' from '{module_name}': {str(e)}"
                    logger.error(msg)
                    failures.append(msg)
                    continue

            try:
                model_cls = get_model_class(name)
            except Exception as e:
                msg = f"Model '{name}' is not registered correctly: {str(e)}"
                logger.error(msg)
                failures.append(msg)
                continue

            device = gpu_manager.acquire()
            logger.info(f"Assigned {name} to device {device}")

            try:
                model = model_cls(device=device)
                models.append(model)
                logger.info(f"{name} created successfully")
            except Exception as e:
                msg = f"Failed to initialize model '{name}' on {device}: {str(e)}"
                logger.error(msg, exc_info=True)
                failures.append(msg)
                continue

        if failures:
            logger.warning(
                "Some models failed to initialize and were skipped: "
                + " | ".join(failures)
            )

        if not models:
            raise RuntimeError(
                "No ASR models were initialized successfully. "
                + " ; ".join(failures)
            )

        logger.info(f"All {len(models)} models created")
        return models
