"""OCR via RapidOCR on ONNX Runtime, NPU-aware.

RapidOCR ships PP-OCRv4 ONNX models and runs them through ONNX Runtime. The
stock wrapper only exposes CPU / CUDA / DirectML, so we inject a small session
subclass that re-orders execution providers to prefer an NPU accelerator (e.g.
Qualcomm's QNN) before falling back to CPU. It always prints which backend it
actually used and never fabricates an NPU claim.
"""
from __future__ import annotations

import logging
import platform
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .utils import NPU_PROVIDERS, detect_onnx_providers, setup_logging

log = setup_logging(__name__)

# QNN is the Qualcomm NPU provider; keep the exact strings we recognize.
_QNN = "QNNExecutionProvider"
_KNOWN_PRIORITY = (_QNN, "DmlExecutionProvider", "CUDAExecutionProvider")


@dataclass
class TextLine:
    """A single recognised text region: text, confidence and box."""

    text: str
    conf: float
    box: list


class OCR:
    """Thin wrapper over RapidOCR with provider injection and graceful failure."""

    def __init__(self, use_npu: bool = True, config_path: Optional[str] = None):
        self.backend = "CPU"
        self.available = False
        self._ocr = None
        self._providers: List[str] = []
        try:
            self._providers, auto_primary = detect_onnx_providers()
            self.backend = "NPU" if (use_npu and auto_primary == "NPU") else "CPU"
            self._install_provider_patch()
            from rapidocr_onnxruntime import RapidOCR  # local import: slow

            kwargs = {"print_verbose": False}
            if config_path:
                kwargs["config_path"] = config_path
            self._ocr = RapidOCR(**kwargs)
            self.available = True
            log.info("OCR ready. Backend=%s providers=%s", self.backend, self._providers[:4])
        except Exception as exc:  # pragma: no cover - env dependant
            log.warning("OCR unavailable (%s). Alerts/Explain will degrade.", exc)
            self.available = False

    @staticmethod
    def _install_provider_patch() -> None:
        """Inject the NPU-aware session into RapidOCR's three components."""
        try:
            from rapidocr_onnxruntime.ch_ppocr_cls import text_cls
            from rapidocr_onnxruntime.ch_ppocr_det import text_detect
            from rapidocr_onnxruntime.ch_ppocr_rec import text_recognize
            from rapidocr_onnxruntime.utils.infer_engine import OrtInferSession

            class ProviderAwareSession(OrtInferSession):  # noqa: D101
                """Provider order prefers an NPU accelerator, then CPU."""

                def _get_ep_list(self) -> List[Tuple[str, dict]]:
                    import onnxruntime as ort  # type: ignore

                    available = set(ort.get_available_providers())
                    ep: List[Tuple[str, dict]] = []
                    used: set = set()
                    for name in _KNOWN_PRIORITY:
                        if name in available and name not in used:
                            ep.append((name, {"device_id": 0}))
                            used.add(name)
                    ep.append(("CPUExecutionProvider",
                               {"arena_extend_strategy": "kSameAsRequested"}))
                    # RapidOCR reads these flags in _verify_providers(); keep the
                    # NPU-aware patch honest so it never claims an unavailable EP.
                    self.use_cuda = "CUDAExecutionProvider" in used
                    self.use_directml = "DmlExecutionProvider" in used
                    return ep

            for mod in (text_detect, text_cls, text_recognize):
                mod.OrtInferSession = ProviderAwareSession
        except Exception as exc:  # pragma: no cover
            log.warning("Could not inject NPU-aware session (%s); using stock backend.", exc)

    def run(self, image: Image.Image) -> List[TextLine]:
        """Run OCR on a PIL image, returning recognised text lines."""
        if not self.available or self._ocr is None:
            return []
        img_np = np.array(image.convert("RGB"))
        try:
            result, _elapse = self._ocr(img_np)
        except Exception as exc:  # pragma: no cover
            log.warning("OCR inference failed: %s", exc)
            return []
        lines: List[TextLine] = []
        if result:
            for item in result:
                try:
                    box, text, conf = item[0], item[1], item[2]
                    if isinstance(text, str) and text.strip():
                        lines.append(TextLine(text=text.strip(), conf=float(conf), box=box))
                except (IndexError, TypeError, ValueError):
                    continue
        return lines


def summarize(text_lines: Sequence[TextLine], limit: int = 2000) -> str:
    """Flatten OCR lines into a bounded string of visible screen text."""
    chunks: List[str] = []
    total = 0
    for line in text_lines:
        if total >= limit:
            break
        chunks.append(line.text)
        total += len(line.text)
    return "\n".join(chunks)