"""CLI runner for single file and batch ASR tasks."""
import os
from typing import List, Optional

from app.lifecycle import app_state, initialize_system
from core.logger import get_logger

logger = get_logger("runner")

SUPPORTED_EXTS = (".wav", ".mp3", ".m4a", ".flac")


class Runner:
    def __init__(self, model_names: List[str], devices: Optional[List[str]] = None):
        logger.info(f"Runner initialized with models: {model_names}, devices: {devices}")
        initialize_system(model_names=model_names, devices=devices)

    def run_single(self, audio_path: str):
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"{audio_path} not found.")

        if app_state.pipeline is None:
            raise RuntimeError("Pipeline is not initialized")

        logger.info(f"Processing single file: {audio_path}")
        result = app_state.pipeline.run_file(audio_path)
        logger.info(f"Completed: {audio_path}")
        return result

    def run_batch(self, folder_path: str):
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"{folder_path} is not a directory.")

        logger.info(f"Batch processing folder: {folder_path}")

        results = []
        file_count = 0

        for root, _, files in os.walk(folder_path):
            for filename in files:
                if not filename.lower().endswith(SUPPORTED_EXTS):
                    continue

                full_path = os.path.join(root, filename)
                try:
                    result = self.run_single(full_path)
                    if isinstance(result, list):
                        results.extend(result)
                    else:
                        results.append(result)
                    file_count += 1
                except Exception as e:
                    logger.error(f"Failed to process {full_path}: {str(e)}")
                    results.append({"source_path": full_path, "path": None, "error": str(e)})

        logger.info(f"Batch processing completed. Processed {file_count} files")
        return results
