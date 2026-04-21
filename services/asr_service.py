from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
import tempfile

from core.logger import get_logger

logger = get_logger("asr_service")


class ASRService:

    def __init__(self, models: List, max_workers: int = 4):
        self.models = models
        self.max_workers = max_workers

    def _infer_model(self, model, audio_path: str):
        try:
            text = model.infer(audio_path)
            return {
                "model": model.model_name,
                "text": text,
                "error": None
            }
        except Exception as e:
            logger.error(f"{model.model_name} failed: {str(e)}")
            return {
                "model": model.model_name,
                "text": None,
                "error": str(e)
            }

    def infer_file(self, audio_path: str):
        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._infer_model, m, audio_path)
                for m in self.models
            ]

            for future in as_completed(futures):
                results.append(future.result())

        return results

    def infer_bytes(self, audio_bytes: bytes):
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            return self.infer_file(tmp.name)
