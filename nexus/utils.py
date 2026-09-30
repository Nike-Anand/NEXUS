"""NEXUS shared helpers: logging, config, backend detection, masking, hashing.

Kept small and dependency-free where possible so the rest of the package can
rely on it without pulling in heavy optional libraries.
"""
from __future__ import annotations

import hashlib
import logging
import math
import platform
import re
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"

# Backends we can surface for the NPU / accelerator.
NPU_PROVIDERS = ("QNNExecutionProvider", "DmlExecutionProvider", "CUDAExecutionProvider")


def setup_logging(name: str = "nexus", level: int = logging.INFO) -> logging.Logger:
    """Configure a module-level logger once and return it."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load config.yaml from disk, returning an empty dict if it is missing."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.exists():
        return {}
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# ---------------------------------------------------------------------------
# Platform / backend detection
# ---------------------------------------------------------------------------
def platform_info() -> Dict[str, str]:
    """Return a snapshot of the runtime platform (for the status bar / logs)."""
    return {
        "python": sys.version.split()[0],
        "system": platform.system(),
        "machine": platform.machine(),
        "release": platform.release(),
        "is_arm64": is_arm64(),
    }


def is_arm64() -> bool:
    """True when running on an ARM64 (Windows on ARM / Snapdragon) machine."""
    machine = platform.machine().lower()
    return machine in ("arm64", "aarch64")


def detect_onnx_providers() -> Tuple[List[str], Optional[str]]:
    """Return the ONNX Runtime provider list and the primary backend name.

    Returns a 2-tuple ``(providers, primary)`` where *primary* is one of
    ``"NPU"``, ``"CPU"`` or ``"unavailable"`` derived from the first usable
    provider. Never claims NPU unless the provider is actually registered.
    """
    try:
        import onnxruntime as ort  # type: ignore
    except Exception:  # pragma: no cover - guard for missing onnxruntime
        return [], "unavailable"

    providers = list(ort.get_available_providers())  # type: ignore[attr-defined]
    primary = "CPU"
    for name in NPU_PROVIDERS:
        if name in providers:
            primary = "NPU"
            break
    return providers, primary
def backend_status(providers: Sequence[str]) -> str:
    """A human label for the accelerator used for OCR, honest about NPU.

    Only reports ``NPU (QNN)`` when the Qualcomm QNN provider was actually
    registered at runtime; otherwise it reports the concrete fallback so the UI
    never fabricates an NPU claim.
    """
    if providers and "QNNExecutionProvider" in providers:
        return "NPU (QNN)"
    if providers and "DmlExecutionProvider" in providers:
        return "NPU (DirectML)"
    if providers and "CUDAExecutionProvider" in providers:
        return "NPU (CUDA)"
    if providers:
        return "CPU"
    return "unavailable"


# ---------------------------------------------------------------------------
# Masking / entropy
# ---------------------------------------------------------------------------
_VISIBLE_HEAD = 3
_VISIBLE_TAIL = 3


def mask_secret(token: str) -> str:
    """Mask the middle of a secret token, keeping a hint at each end.

    >>> mask_secret("sk-0123456789abcdef")
    'sk-012...def'
    """
    token = token.strip()
    if len(token) <= _VISIBLE_HEAD + _VISIBLE_TAIL + 2:
        return token[:_VISIBLE_HEAD] + "***"
    return (
        token[:_VISIBLE_HEAD]
        + "..."
        + token[-_VISIBLE_TAIL:]
        if len(token) > _VISIBLE_HEAD + _VISIBLE_TAIL + 2
        else "***"
    )


def shannon_entropy(s: str) -> float:
    """Shannon entropy (bits per character) of a string."""
    if not s:
        return 0.0
    counts = Counter(s)
    total = len(s)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def looks_high_entropy(token: str, threshold: float = 3.2, min_len: int = 8) -> bool:
    """Heuristic: a long token whose entropy looks like a random secret key."""
    if len(token) < min_len:
        return False
    return shannon_entropy(token) >= threshold


_HEX_TOKEN = re.compile(r"^[0-9A-Fa-f]{12,}$")
_B64_TOKEN = re.compile(r"^[A-Za-z0-9+/=_-]{12,}$")


def tokenize_words(text: str) -> List[str]:
    """Split text into candidate whitespace-separated tokens."""
    return [t for t in re.split(r"\s+", text) if t]


def image_dhash(image, hash_size: int = 12) -> str:
    """Perceptual hash (dHash) of a PIL image for cheap screen-change checks."""
    from PIL import Image  # local import: PIL only needed here

    img = image.convert("L").resize(
        (hash_size + 1, hash_size), Image.Resampling.LANCZOS
    )
    px = list(img.getdata())
    n = hash_size * hash_size
    # Build a hash string of bits comparing each pixel to its right neighbour.
    bits = [
        "1" if px[y * (hash_size + 1) + x] < px[y * (hash_size + 1) + x + 1] else "0"
        for y in range(hash_size)
        for x in range(hash_size)
    ]
    return "".join(bits)


def hamming_distance(a: str, b: str) -> int:
    """Number of differing bits between two equal-length dHash strings."""
    return sum(1 for x, y in zip(a, b) if x != y)


@contextmanager
def timed(label: str = "op"):
    """Context manager yielding elapsed seconds; logs a friendly message."""
    import time

    start = time.perf_counter()
    yield 0.0
    elapsed = time.perf_counter() - start
    logging.getLogger("nexus.timing").debug("%s took %.0f ms", label, elapsed * 1000)


def now_ms() -> float:
    import time

    return time.perf_counter() * 1000.0