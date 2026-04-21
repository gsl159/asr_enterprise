"""Speaker matching service based on reference voice library."""
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import librosa
import numpy as np

from core.logger import get_logger

logger = get_logger("speaker_service")

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


class SpeakerService:
    def __init__(self, runtime_config: Optional[Dict[str, Any]] = None, enabled: bool = False):
        config = runtime_config if isinstance(runtime_config, dict) else {}
        speaker_cfg = config.get("speaker", {})
        if not isinstance(speaker_cfg, dict):
            speaker_cfg = {}

        self.enabled = _as_bool(speaker_cfg.get("enabled", enabled), default=enabled)
        self.sample_rate = max(1, _as_int(speaker_cfg.get("sample_rate", 16000), 16000))
        self.min_score = float(speaker_cfg.get("min_score", 0.35))

        library_dir = (
            speaker_cfg.get("library_dir")
            or os.getenv("ASR_SPEAKER_LIBRARY_DIR")
        )
        self.library_dir = Path(library_dir).expanduser() if library_dir else None

        self.references: List[Dict[str, Any]] = []
        if self.enabled:
            self._load_reference_library()
        else:
            logger.warning("Speaker recognition disabled, fallbacking to unknown speaker")

    def _base_unknown(self, audio_path: str) -> Dict[str, Any]:
        return {
            "path": audio_path,
            "speaker_name": "unknown",
            "speaker_wav": None,
            "speaker_id": "speaker_000",
            "speaker_score": 0.0,
        }

    def _extract_embedding(self, audio_path: str) -> np.ndarray:
        audio, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        if audio.size == 0:
            raise RuntimeError(f"empty audio: {audio_path}")

        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=20)
        delta = librosa.feature.delta(mfcc)
        feature = np.concatenate(
            [
                np.mean(mfcc, axis=1),
                np.std(mfcc, axis=1),
                np.mean(delta, axis=1),
                np.std(delta, axis=1),
            ]
        ).astype(np.float32)

        norm = float(np.linalg.norm(feature))
        if norm <= 1e-12:
            return feature
        return feature / norm

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 1e-12:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _load_reference_library(self):
        if self.library_dir is None:
            logger.warning("Speaker library dir is not configured")
            return
        if not self.library_dir.exists():
            logger.warning(f"Speaker library dir not found: {self.library_dir}")
            return

        speaker_idx = 0
        for path in sorted(self.library_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_AUDIO_EXTS:
                continue
            try:
                embedding = self._extract_embedding(str(path))
            except Exception as e:
                logger.warning(f"Failed to load speaker reference '{path}': {str(e)}")
                continue

            speaker_idx += 1
            self.references.append(
                {
                    "speaker_name": path.stem,
                    "speaker_wav": str(path),
                    "speaker_id": f"speaker_{speaker_idx:03d}",
                    "embedding": embedding,
                }
            )

        logger.info(f"Speaker references loaded: {len(self.references)}")

    def run(self, audio_path: str):
        unknown = self._base_unknown(audio_path)
        if not self.enabled:
            return unknown
        if not self.references:
            return unknown

        try:
            query = self._extract_embedding(audio_path)
        except Exception as e:
            logger.warning(f"Speaker embedding failed for {audio_path}: {str(e)}")
            return unknown

        best = None
        best_score = -1.0
        for ref in self.references:
            score = self._cosine_similarity(query, ref["embedding"])
            if score > best_score:
                best_score = score
                best = ref

        if best is None or best_score < self.min_score:
            return unknown

        return {
            "path": audio_path,
            "speaker_name": best["speaker_name"],
            "speaker_wav": best["speaker_wav"],
            "speaker_id": best["speaker_id"],
            "speaker_score": round(float(best_score), 6),
        }
