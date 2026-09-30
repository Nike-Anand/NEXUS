"""Environment / dependency smoke test for NEXUS.

Run:  python scripts/check_env.py

Verifies packages, prints Windows ARM64 info, ONNX Runtime providers, whether
the NPU (QNN) is usable, and the selected LLM backend. Exits non-zero if a
*required* dependency is missing; optional ones are reported but not fatal.
"""
from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexus.llm import _detect_backend  # noqa: E402
from nexus.utils import detect_onnx_providers, is_arm64, platform_info  # noqa: E402

REQUIRED = ["numpy", "PIL", "yaml", "mss", "cv2", "onnxruntime", "rapidocr_onnxruntime"]
OPTIONAL = {
    "ollama": ("LLM backend (Ollama)", True),
    "onnxruntime_genai": ("NPU LLM (Qualcomm genai)", False),
    "llama_cpp": ("LLM backend (llama-cpp GGUF)", True),
    "matplotlib": ("benchmark charting", True),
}


def banner(title: str) -> None:
    print("\n" + "=" * 64)
    print(f"  {title}")
    print("=" * 64)


def main() -> int:
    banner("NEXUS environment check")
    ok = True

    print(f"Python        : {platform.python_version()}")
    print(f"Machine       : {platform.machine()}  (ARM64: {is_arm64()})")
    print(f"OS            : {platform.system()} {platform.release()}")
    print(f"  {platform_info()}")

    banner("Required packages")
    for name in REQUIRED:
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, "__version__", "?")
            print(f"  [ok] {name:22} {version}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"  [MISSING] {name:16} -> {exc}")

    banner("Optional packages")
    for name, (label, is_backend) in OPTIONAL.items():
        try:
            importlib.import_module(name)
            print(f"  [ok]   {label:30} ({name})")
        except Exception:  # noqa: BLE001
            state = "recommended" if is_backend else "only for NPU LLM"
            print(f"  [skip] {label:30} not installed ({state})")

    banner("ONNX Runtime & accelerator")
    providers, primary = detect_onnx_providers()
    print(f"  Providers    : {providers}")
    print(f"  Primary      : {primary}")
    if "QNNExecutionProvider" in providers:
        print("  NPU (QNN)    : [AVAILABLE] Qualcomm NPU usable")
    elif primary in ("Dml", "CUDA"):
        print("  NPU (QNN)    : not found; using " + primary)
    else:
        print("  NPU (QNN)    : not present on this machine -> CPU path (honest)")

    banner("LLM backend chain")
    try:
        print(f"  Detected backend : {_detect_backend()}")
        print("  (install ollama / onnxruntime-genai / llama-cpp for a real LLM)")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"  {exc}")

    print("\n" + "=" * 64)
    if ok:
        print("  RESULT: PASS - core dependencies are ready.")
    else:
        print("  RESULT: FAIL - install missing required packages (see requirements.txt).")
    print("=" * 64)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())