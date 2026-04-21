"""Dependency checker for the ASR project.

This script only inspects the current environment and does not install packages.
"""
import importlib
import re
import sys
from importlib import metadata

REQUIRED_PACKAGES = {
    # import_name: minimal_version
    "fastapi": "0.110.0",
    "uvicorn": "0.27.1",
    "pydantic": "2.6.1",
    "aiofiles": "23.2.1",
    "packaging": "23.0",
    "yaml": "6.0.1",  # provided by PyYAML
    "torch": "2.8.0",
    "torchaudio": "2.8.0",
    "torchvision": "0.23.0",
    "whisper": "20231117",  # provided by openai-whisper
    "funasr": "1.0.20",
    "librosa": "0.10.1",
    "soundfile": "0.12.1",
    "numpy": "1.24.0",
    "scipy": "1.10.0",
}

OPTIONAL_PACKAGES = {
    "dotenv": "1.0.0",  # provided by python-dotenv
    "pythonjsonlogger": "2.0.7",  # provided by python-json-logger
    "gunicorn": "21.2.0",
}

# import_name -> distribution_name (when they differ)
DIST_NAME_MAP = {
    "yaml": "PyYAML",
    "whisper": "openai-whisper",
    "dotenv": "python-dotenv",
    "pythonjsonlogger": "python-json-logger",
}


def _version_tuple(version: str):
    # Keep numeric parts only for a best-effort comparison.
    tokens = re.findall(r"\d+", version or "")
    return tuple(int(t) for t in tokens)


def _is_version_ok(actual: str, minimum: str) -> bool:
    if not minimum:
        return True
    if not actual:
        return False
    return _version_tuple(actual) >= _version_tuple(minimum)


def _get_distribution_version(import_name: str):
    dist_name = DIST_NAME_MAP.get(import_name, import_name)
    try:
        return metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        return None


def check_package(import_name: str, min_version: str = None):
    result = {
        "installed": False,
        "version": None,
        "required_version": min_version,
        "meets_requirement": False,
        "error": None,
    }

    try:
        module = importlib.import_module(import_name)
        result["installed"] = True

        module_version = getattr(module, "__version__", None)
        dist_version = _get_distribution_version(import_name)
        actual_version = module_version or dist_version
        result["version"] = actual_version

        if _is_version_ok(actual_version, min_version):
            result["meets_requirement"] = True
        else:
            result["error"] = f"version too old (need >= {min_version}, actual {actual_version})"
    except Exception as e:
        result["error"] = str(e)

    return result


def print_result(name: str, result: dict, required: bool):
    status = "OK" if result["meets_requirement"] else "MISSING"
    scope = "required" if required else "optional"
    version = result["version"] or "-"
    error = result["error"] or ""
    print(f"[{status}] {name:18} ({scope:8}) version={version:12} {error}")


def check_gpu():
    try:
        import torch
    except Exception as e:
        return False, [f"torch import failed: {e}"]

    issues = []

    if not torch.cuda.is_available():
        issues.append("CUDA is not available")
        return False, issues

    try:
        device_name = torch.cuda.get_device_name(0)
        cuda_version = getattr(torch.version, "cuda", None)
        print(f"\nGPU: {device_name}")
        print(f"CUDA: {cuda_version}")
    except Exception as e:
        issues.append(f"failed to query GPU info: {e}")

    return len(issues) == 0, issues


def main():
    print("=" * 80)
    print("ASR Environment Check")
    print("=" * 80)

    required_failures = []

    print("\nRequired packages")
    for name, min_ver in REQUIRED_PACKAGES.items():
        result = check_package(name, min_ver)
        print_result(name, result, required=True)
        if not result["meets_requirement"]:
            required_failures.append((name, result["error"]))

    print("\nOptional packages")
    for name, min_ver in OPTIONAL_PACKAGES.items():
        result = check_package(name, min_ver)
        print_result(name, result, required=False)

    gpu_ok, gpu_issues = check_gpu()
    if gpu_issues:
        print("\nGPU issues")
        for issue in gpu_issues:
            print(f"- {issue}")

    print("\n" + "=" * 80)
    if not required_failures and gpu_ok:
        print("Environment check passed")
        print("Start command: uvicorn app.main:app --host 0.0.0.0 --port 8000")
        return 0

    print("Environment check failed")
    if required_failures:
        print("\nMissing or invalid required packages")
        for pkg, err in required_failures:
            print(f"- {pkg}: {err}")
    print("\nInstall command: pip install -r requirements.txt")
    return 1


if __name__ == "__main__":
    sys.exit(main())
