import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from app.runner import Runner
from core.logger import get_logger
from services.result_service import ResultService

logger = get_logger("cli")

AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".flac")


def _flatten_records(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        flattened: List[Dict[str, Any]] = []
        for item in payload:
            flattened.extend(_flatten_records(item))
        return flattened
    return []


def _write_json(output_path: Path, records: List[Dict[str, Any]]):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def _write_csv(output_path: Path, records: List[Dict[str, Any]]):
    rows = [ResultService.to_csv_ready(record) for record in records]
    if not rows:
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_output(output: str, records: List[Dict[str, Any]]):
    output_path = Path(output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        _write_csv(output_path, records)
    else:
        _write_json(output_path, records)
    logger.info(f"Saved {len(records)} records to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Enterprise ASR CLI")

    parser.add_argument("--input", required=True, help="Input audio file or folder")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "paraformer_qf",
            "whisper_turbo",
            "paraformer_zh",
            "sensevoice",
            "wenet_speech",
            "funasr_nano",
        ],
        help="Models to load",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=None,
        help="GPU devices, e.g., cuda:0 cuda:1",
    )
    parser.add_argument("--output", default=None, help="Output file (.json or .csv)")

    args = parser.parse_args()

    runner = Runner(model_names=args.models, devices=args.devices)
    if args.input.lower().endswith(AUDIO_EXTS):
        result = runner.run_single(args.input)
    else:
        result = runner.run_batch(args.input)

    records = _flatten_records(result)
    if args.output:
        _write_output(args.output, records)
    else:
        print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
