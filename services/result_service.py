import json
from typing import Any, Dict, List, Optional


class ResultService:
    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def build_result(
        self,
        segment_record: Dict[str, Any],
        std_asr_text: str,
        asr_score: float,
        asr_model_scores: Dict[str, float],
        raw_results: List[dict],
        speaker: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        record = dict(segment_record) if isinstance(segment_record, dict) else {}
        path = record.get("path")

        speaker_info = speaker if isinstance(speaker, dict) else {}
        speaker_name = speaker_info.get("speaker_name", "unknown")
        speaker_wav = speaker_info.get("speaker_wav")
        speaker_id = speaker_info.get("speaker_id", "speaker_000")
        speaker_score = self._safe_float(speaker_info.get("speaker_score", 0.0), default=0.0)

        result = {
            **record,
            "path": path,
            "std_asr_text": std_asr_text,
            "asr_score": self._safe_float(asr_score, default=1.0),
            "asr_model_scores": asr_model_scores or {},
            "asr_models": raw_results,
            "speaker_name": speaker_name,
            "speaker_wav": speaker_wav,
            "speaker_id": speaker_id,
            "speaker_score": speaker_score,
        }

        # Backward-compatible fields for old consumers.
        result["audio"] = path
        result["text"] = std_asr_text
        result["confidence"] = round(max(0.0, 1.0 - result["asr_score"]), 6)
        result["speaker"] = {
            "speaker_name": speaker_name,
            "speaker_wav": speaker_wav,
            "speaker_id": speaker_id,
            "speaker_score": speaker_score,
        }
        return result

    def build_error_result(self, segment_record: Dict[str, Any], error_message: str) -> Dict[str, Any]:
        record = dict(segment_record) if isinstance(segment_record, dict) else {}
        path = record.get("path")
        return {
            **record,
            "path": path,
            "std_asr_text": "",
            "asr_score": 1.0,
            "asr_model_scores": {},
            "asr_models": [],
            "speaker_name": "unknown",
            "speaker_wav": None,
            "speaker_id": "speaker_000",
            "speaker_score": 0.0,
            "error": error_message,
            "audio": path,
            "text": "",
            "confidence": 0.0,
            "speaker": {
                "speaker_name": "unknown",
                "speaker_wav": None,
                "speaker_id": "speaker_000",
                "speaker_score": 0.0,
            },
        }

    @staticmethod
    def to_csv_ready(record: Dict[str, Any]) -> Dict[str, Any]:
        row = {}
        for key, value in record.items():
            if isinstance(value, (dict, list)):
                row[key] = json.dumps(value, ensure_ascii=False)
            else:
                row[key] = value
        return row
