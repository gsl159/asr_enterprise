"""Inference pipeline orchestrator (VAD -> ASR -> Speaker -> merged output)."""
import threading
import time
from queue import Empty
from typing import Any, Dict, List, Optional

from core.logger import get_logger
from services.aggregate_service import AggregateService
from services.result_service import ResultService
from services.speaker_service import SpeakerService
from services.vad_service import VADService

logger = get_logger("pipeline")


class PipelineOrchestrator:
    """ASR main orchestrator."""

    def __init__(self, worker_manager, runtime_config: Optional[Dict[str, Any]] = None):
        if worker_manager is None:
            raise ValueError("worker_manager must not be None")
        self.worker_manager = worker_manager
        self.runtime_config = runtime_config if isinstance(runtime_config, dict) else {}

        self.vad_service = VADService(runtime_config=self.runtime_config)
        self.aggregate_service = AggregateService()
        self.speaker_service = SpeakerService(runtime_config=self.runtime_config)
        self.result_service = ResultService()

        self.results_dict: Dict[int, Any] = {}
        self.result_events: Dict[int, threading.Event] = {}
        self.result_lock = threading.Lock()

        self.collector_thread = threading.Thread(target=self._collect_results, daemon=True)
        self.collector_thread.start()

        logger.info("Pipeline orchestrator initialized")

    def _collect_results(self):
        consecutive_errors = 0
        while True:
            try:
                task_id, result = self.worker_manager.get_result(timeout=5.0)
            except Empty:
                if getattr(self.worker_manager, "stopped", None) and self.worker_manager.stopped.is_set():
                    logger.info("Result collector exiting after worker manager shutdown")
                    break
                consecutive_errors = 0
                continue
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Result collector error: {str(e)}")
                if consecutive_errors > 10:
                    logger.critical("Result collector exiting due to repeated errors")
                    break
                time.sleep(1)
                continue

            with self.result_lock:
                self.results_dict[task_id] = result
                event = self.result_events.get(task_id)
                if event is not None:
                    event.set()
            consecutive_errors = 0

    def _wait_for_result(self, task_id: int, timeout: int = 300):
        if not self.collector_thread.is_alive():
            raise RuntimeError("Result collector thread is not running")

        with self.result_lock:
            if task_id in self.results_dict:
                self.result_events.pop(task_id, None)
                return self.results_dict.pop(task_id)
            event = threading.Event()
            self.result_events[task_id] = event

        if event.wait(timeout=timeout):
            with self.result_lock:
                self.result_events.pop(task_id, None)
                result = self.results_dict.pop(task_id, None)
            if result is None:
                raise RuntimeError(f"Task {task_id} signaled but result payload missing")
            return result

        with self.result_lock:
            self.result_events.pop(task_id, None)
        raise TimeoutError(f"Task {task_id} timeout after {timeout}s")

    def run_file(self, audio_path: str, source_path: Optional[str] = None) -> List[Dict[str, Any]]:
        source_ref = source_path or audio_path
        logger.info(f"========== Pipeline Start: {source_ref} ==========")
        segments = self.vad_service.run(audio_path, source_path=source_path)
        if not isinstance(segments, list):
            raise RuntimeError("VAD service must return a list of segment records")
        if not segments:
            logger.warning(f"No segments produced by VAD: {source_ref}")
            return [
                self.result_service.build_error_result(
                    {"path": None, "source_path": source_ref},
                    "No speech detected by VAD",
                )
            ]

        logger.info(f"VAD output segments: {len(segments)}")
        final_outputs: List[Dict[str, Any]] = []

        for idx, seg in enumerate(segments):
            if not isinstance(seg, dict):
                final_outputs.append(self.result_service.build_error_result({}, "Invalid segment payload"))
                continue

            seg_path = seg.get("path")
            if not isinstance(seg_path, str) or not seg_path:
                final_outputs.append(self.result_service.build_error_result(seg, "Invalid segment path"))
                continue

            logger.info(f"Processing segment {idx + 1}/{len(segments)}: {seg_path}")
            task_ids = self.worker_manager.submit_to_all(seg_path)

            merged_results = []
            segment_errors = []
            for task_id in task_ids:
                raw_results = self._wait_for_result(task_id)
                if isinstance(raw_results, dict) and "error" in raw_results:
                    segment_errors.append(str(raw_results["error"]))
                    continue
                if not isinstance(raw_results, list):
                    segment_errors.append("Invalid model output payload")
                    continue
                merged_results.extend(raw_results)

            if not merged_results:
                err_msg = " ; ".join(segment_errors) if segment_errors else "No model outputs"
                final_outputs.append(self.result_service.build_error_result(seg, err_msg))
                continue

            std_asr_text, asr_score, asr_model_scores = self.aggregate_service.aggregate(merged_results)
            speaker_result = self.speaker_service.run(seg_path)

            merged = self.result_service.build_result(
                segment_record=seg,
                std_asr_text=std_asr_text,
                asr_score=asr_score,
                asr_model_scores=asr_model_scores,
                raw_results=merged_results,
                speaker=speaker_result,
            )
            if segment_errors:
                merged["partial_errors"] = segment_errors
            final_outputs.append(merged)

        logger.info(f"========== Pipeline Complete: outputs={len(final_outputs)} ==========")
        return final_outputs
