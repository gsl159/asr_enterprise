"""VAD service with physical slicing output."""
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import librosa
import soundfile as sf
import torch
from funasr import AutoModel

from core.logger import get_logger

logger = get_logger("vad_service")

DEFAULT_VAD_MODEL_PATH = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
SUPPORTED_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac"}


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class VADService:
    def __init__(self, runtime_config: Optional[Dict[str, Any]] = None):
        config = runtime_config if isinstance(runtime_config, dict) else {}
        vad_cfg = config.get("vad", {})
        output_cfg = config.get("output", {})
        if not isinstance(vad_cfg, dict):
            vad_cfg = {}
        if not isinstance(output_cfg, dict):
            output_cfg = {}

        self.enabled = _as_bool(vad_cfg.get("enabled", True), default=True)
        self.sample_rate = max(1, _as_int(vad_cfg.get("sample_rate", 16000), 16000))
        self.pad_ms = max(0, _as_int(vad_cfg.get("pad_ms", 200), 200))
        self.merge_gap_ms = max(0, _as_int(vad_cfg.get("merge_gap_ms", 300), 300))
        self.min_seg_ms = max(1, _as_int(vad_cfg.get("min_seg_ms", 500), 500))
        self.max_seg_ms = max(self.min_seg_ms, _as_int(vad_cfg.get("max_seg_ms", 10000), 10000))

        base_output_dir = vad_cfg.get("output_dir") or output_cfg.get("output_dir") or "./outputs"
        self.output_root = Path(base_output_dir).expanduser() / "vad_packs"
        self.output_root.mkdir(parents=True, exist_ok=True)

        self.model_path = (
            vad_cfg.get("model_path")
            or os.getenv("ASR_VAD_MODEL_PATH")
            or DEFAULT_VAD_MODEL_PATH
        )

        self._lock = threading.Lock()
        self._pack_counter = self._detect_existing_pack_counter()
        self._global_id = 0

        self.vad_model = None
        if self.enabled:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading VAD model on {device}: {self.model_path}")
            self.vad_model = AutoModel(model=self.model_path, device=device)
        else:
            logger.warning("VAD is disabled in config, using whole-audio fallback segmentation")

    def _detect_existing_pack_counter(self) -> int:
        max_pack = 0
        for path in self.output_root.glob("pack_*"):
            if not path.is_dir():
                continue
            try:
                max_pack = max(max_pack, int(path.name.split("_", 1)[1]))
            except (IndexError, ValueError):
                continue
        return max_pack

    def _next_pack_id(self) -> int:
        with self._lock:
            self._pack_counter += 1
            return self._pack_counter

    def _next_segment_id(self) -> int:
        with self._lock:
            self._global_id += 1
            return self._global_id

    @staticmethod
    def _merge_segments(segments: Sequence[Tuple[int, int]], merge_gap_ms: int) -> List[Tuple[int, int]]:
        if not segments:
            return []
        sorted_segments = sorted(segments, key=lambda item: item[0])
        merged = [list(sorted_segments[0])]
        for start_ms, end_ms in sorted_segments[1:]:
            prev_start, prev_end = merged[-1]
            if start_ms - prev_end <= merge_gap_ms:
                merged[-1][1] = max(prev_end, end_ms)
            else:
                merged.append([start_ms, end_ms])
        return [(start, end) for start, end in merged]

    @staticmethod
    def _enforce_length_rules(
        segments: Iterable[Tuple[int, int]],
        min_ms: int,
        max_ms: int,
    ) -> List[Tuple[int, int]]:
        final_segments: List[Tuple[int, int]] = []
        for start_ms, end_ms in segments:
            duration = end_ms - start_ms
            if duration < min_ms:
                continue
            if duration <= max_ms:
                final_segments.append((start_ms, end_ms))
                continue
            cursor = start_ms
            while cursor < end_ms:
                chunk_end = min(cursor + max_ms, end_ms)
                if chunk_end - cursor >= min_ms:
                    final_segments.append((cursor, chunk_end))
                cursor = chunk_end
        return final_segments

    def _preprocess_audio(self, audio_path: str):
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        if audio.size == 0:
            raise RuntimeError(f"Empty audio data: {audio_path}")
        return audio, sr

    @staticmethod
    def _extract_segments(raw_result: Any) -> List[Tuple[int, int]]:
        if isinstance(raw_result, list) and raw_result:
            payload = raw_result[0]
        else:
            payload = raw_result

        if isinstance(payload, dict):
            candidates = (
                payload.get("value")
                or payload.get("segments")
                or payload.get("timestamp")
                or payload.get("timestamps")
                or []
            )
        else:
            candidates = []

        final: List[Tuple[int, int]] = []
        for item in candidates:
            start_ms = None
            end_ms = None
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                start_ms, end_ms = item[0], item[1]
            elif isinstance(item, dict):
                start_ms = item.get("start", item.get("begin"))
                end_ms = item.get("end", item.get("stop"))
            try:
                start_val = int(float(start_ms))
                end_val = int(float(end_ms))
            except (TypeError, ValueError):
                continue
            if end_val > start_val >= 0:
                final.append((start_val, end_val))
        return final

    def _infer_segments(self, audio_path: str, full_audio_ms: int) -> List[Tuple[int, int]]:
        if not self.enabled:
            return [(0, full_audio_ms)]
        if self.vad_model is None:
            raise RuntimeError("VAD model is not initialized")

        raw = self.vad_model.generate(input=audio_path)
        segments = self._extract_segments(raw)
        if not segments:
            return []

        padded = []
        for start_ms, end_ms in segments:
            padded.append((max(0, start_ms - self.pad_ms), end_ms + self.pad_ms))

        merged = self._merge_segments(padded, self.merge_gap_ms)
        clipped = [(max(0, s), min(full_audio_ms, e)) for s, e in merged if e > s]
        return self._enforce_length_rules(clipped, self.min_seg_ms, self.max_seg_ms)

    def _write_pack_jsonl(self, pack_dir: Path, pack_id: int, records: List[Dict[str, Any]]):
        jsonl_path = pack_dir / f"pack_{pack_id:03d}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def run(self, audio_path: str, source_path: Optional[str] = None) -> List[Dict[str, Any]]:
        actual_path = Path(audio_path).expanduser().resolve()
        if not actual_path.exists():
            raise FileNotFoundError(f"Audio file not found: {actual_path}")

        if actual_path.suffix.lower() not in SUPPORTED_AUDIO_EXTS:
            logger.warning(f"Non-standard audio extension for VAD: {actual_path}")

        source_display_path = str(source_path) if source_path else str(actual_path)
        source_label = Path(source_display_path).stem or actual_path.stem

        audio, sr = self._preprocess_audio(str(actual_path))
        full_audio_ms = int(round(len(audio) * 1000 / sr))
        vad_segments = self._infer_segments(str(actual_path), full_audio_ms)
        if not vad_segments:
            logger.warning(f"No speech detected by VAD: {source_display_path}")
            return []

        pack_id = self._next_pack_id()
        pack_dir = self.output_root / f"pack_{pack_id:03d}"
        pack_dir.mkdir(parents=True, exist_ok=True)

        records: List[Dict[str, Any]] = []
        for seg_index, (start_ms, end_ms) in enumerate(vad_segments):
            start_sample = max(0, min(int(round(start_ms * sr / 1000)), len(audio)))
            end_sample = max(start_sample, min(int(round(end_ms * sr / 1000)), len(audio)))
            if end_sample <= start_sample:
                continue

            segment_id = self._next_segment_id()
            seg_path = pack_dir / f"seg_{segment_id:06d}_{source_label}.wav"
            sf.write(seg_path, audio[start_sample:end_sample], sr)

            duration_ms = int(round((end_sample - start_sample) * 1000 / sr))
            record = {
                "id": segment_id,
                "path": str(seg_path),
                "source_path": source_display_path,
                "seg_index": seg_index,
                "start_time_mm": int(start_ms),
                "end_time_mm": int(end_ms),
                "dur_time_mm": duration_ms,
                "dur_time_s": round(duration_ms / 1000.0, 3),
                "rate": int(sr),
                "pack_id": pack_id,
            }
            records.append(record)

        self._write_pack_jsonl(pack_dir, pack_id, records)
        logger.info(
            f"VAD sliced file into {len(records)} segments: source={source_display_path}, pack_id={pack_id}"
        )
        return records
