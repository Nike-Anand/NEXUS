"""Local LLM wrapper with a backend priority chain and automatic fallback.

Priority (first available wins):
  a) onnxruntime-genai / Qualcomm AI Hub model (e.g. Phi-3.5 mini, quantized)
  b) Ollama local server (if installed) with a small model
  c) llama-cpp-python with a small GGUF model
  d) "no-LLM mode": a deterministic template summarizer - fully on-device.

If no backend is available NEXUS still works: Alerts are rule-based and Explain
uses the template path. Nothing here ever makes a network call at runtime.
"""
from __future__ import annotations

import logging
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .utils import setup_logging

log = setup_logging(__name__)

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is",
    "are", "was", "were", "be", "this", "that", "it", "as", "at", "by", "from",
    "you", "your", "we", "our", "i", "me", "my", "not", "no", "but", "if", "then",
    "will", "can", "do", "does", "what", "how", "when", "where", "which", "has",
    "have", "there", "their",
}


@dataclass
class LLMStats:
    """Timing / throughput stats for a single generation."""

    backend: str = ""
    time_to_first_token_ms: float = 0.0
    total_ms: float = 0.0
    tokens: int = 0
    tokens_per_sec: float = 0.0

    def messages(self) -> List[str]:
        return [
            f"backend={self.backend}",
            f"tokens={self.tokens}",
            f"ttft={(self.time_to_first_token_ms / 1_000):.2f}s",
            f"tokens/s={(self.tokens_per_sec):.1f}",
        ]


class _NoLLM:
    """Deterministic template fallback - always available, zero dependencies."""

    name = "template"

    def generate(self, prompt: str, max_tokens: int = 220) -> str:
        return _template_answer(prompt)


# ---------------------------------------------------------------------------
# Backend discovery (lazy imports; each may be absent)
# ---------------------------------------------------------------------------
def _detect_backend(order: Optional[List[str]] = None) -> str:
    order = order or ["onnx", "ollama", "llama_cpp"]
    for name in order:
        if name == "onnx" and _onnx_available():
            return "onnx"
        if name == "ollama" and _ollama_available():
            return "ollama"
        if name == "llama_cpp" and _llama_cpp_available():
            return "llama_cpp"
    return "template"


def _onnx_available() -> bool:
    try:
        import onnxruntime_genai  # noqa: F401
        return True
    except Exception:
        return False


def _ollama_available() -> bool:
    try:
        import ollama  # noqa: F401
        return True
    except Exception:
        return False


def _llama_cpp_available() -> bool:
    try:
        import llama_cpp  # noqa: F401
        return True
    except Exception:
        return False
# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------
class LLM:
    """Thin client that picks a backend at startup and exposes ``generate``."""

    def __init__(self, config: Optional[dict] = None, backend: Optional[str] = None):
        cfg = config or {}
        self.order = cfg.get("backend_order", ["onnx", "ollama", "llama_cpp"])
        self.backend = backend or _detect_backend(self.order)
        self.model = cfg.get("model", "phi-3.5-mini-instruct-q4")
        self.max_tokens = int(cfg.get("max_tokens", 220))
        self._client = self._build_client()
        log.info("LLM backend selected: %s", self.backend)

    def _build_client(self):
        if self.backend == "onnx":
            try:
                from onnxruntime_genai import Model  # type: ignore
                return _OnnxGenAI(Model, self.model)
            except Exception as exc:  # pragma: no cover
                log.warning("onnx runtime failed to start (%s); using template", exc)
        if self.backend == "ollama":
            try:
                import ollama  # type: ignore
                return _Ollama(ollama, self.model)
            except Exception as exc:  # pragma: no cover
                log.warning("ollama client failed (%s); using template", exc)
        if self.backend == "llama_cpp":
            try:
                from llama_cpp import Llama  # type: ignore
                return _LlamaCpp(Llama, self.model)
            except Exception as exc:  # pragma: no cover
                log.warning("llama-cpp failed to start (%s); using template", exc)
        return _NoLLM()

    # -- public API ----------------------------------------------------------
    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> Dict:
        """Return ``{text, stats}``. Never raises: degrades to the template."""
        start = time.perf_counter()
        try:
            text = self._client.generate(prompt, max_tokens or self.max_tokens)
        except Exception as exc:  # pragma: no cover
            log.warning("LLM generation failed (%s); falling back to template", exc)
            text = _template_answer(prompt)
        elapsed_s = time.perf_counter() - start
        tokens = max(1, len(text.split()))
        stats = LLMStats(
            backend=self._client.name,
            total_ms=elapsed_s * 1000.0,
            tokens=tokens,
            tokens_per_sec=tokens / elapsed_s,
        )
        return {"text": text, "stats": stats}

    def stream(self, prompt: str, on_token: Callable[[str], None],
               max_tokens: Optional[int] = None) -> str:
        """Stream generation to ``on_token``; returns the joined text."""
        full: List[str] = []
        for chunk in self._stream_chunks(prompt, max_tokens or self.max_tokens):
            full.append(chunk)
            on_token(chunk)
        return "".join(full)

    def _stream_chunks(self, prompt: str, max_tokens: int) -> List[str]:
        if self._client.name == "template":
            return _chunk(_template_answer(prompt))
        return _chunk(self._client.generate(prompt, max_tokens))


def _chunk(text: str, size: int = 12) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i + size]) + " " for i in range(0, len(words), size)]
# ---------------------------------------------------------------------------
# Concrete backends (thin, best-effort)
# ---------------------------------------------------------------------------
class _OnnxGenAI:
    """Qualcomm onnxruntime-genai NPU backend."""

    name = "onnx-npu"

    def __init__(self, model_cls, config: str):
        self._model = model_cls(config)

    def generate(self, prompt: str, max_tokens: int) -> str:
        return self._model.generate(prompt, max_tokens=max_tokens)


class _Ollama:
    """Local Ollama server backend."""

    name = "ollama"

    def __init__(self, mod, model: str):
        self._ollama = mod
        self.model = model

    def generate(self, prompt: str, max_tokens: int) -> str:
        resp = self._ollama.generate(
            self.model, prompt, options={"num_predict": max_tokens}
        )
        text = getattr(resp, "response", "")
        return str(text) if text else str(resp)


class _LlamaCpp:
    """llama-cpp-python GGUF backend."""

    name = "llama-cpp"

    def __init__(self, llm_cls, config: str):
        self._llama = llm_cls(model_path=config)

    def generate(self, prompt: str, max_tokens: int) -> str:
        out = self._llama(prompt, max_tokens=max_tokens)
        return out["choices"][0]["text"]


# ---------------------------------------------------------------------------
# Template (no-LLM) summarizer
# ---------------------------------------------------------------------------
def _extract_terms(text: str, top: int = 8) -> List[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text.lower())
    words = [w for w in words if w not in _STOPWORDS and not w.isdigit()]
    counts: Counter = Counter(words)
    return [w for w, _ in counts.most_common(top)]


_SCREEN_MARKER = "[SCREEN CONTEXT]\n"


def _template_answer(prompt: str) -> str:
    """Build a deterministic answer from the prompt's embedded screen context."""
    if _SCREEN_MARKER in prompt:
        screen = prompt.split(_SCREEN_MARKER, 1)[1].split("\n\n", 1)[0]
    else:
        screen = prompt
    topics = ", ".join(_extract_terms(screen, 6)) or "unrecognized content"
    notable = _notable_markers(screen)
    text = f"I can see {topics} on screen."
    if notable:
        text += f" Notable: {notable}."
    text += " (Template mode - install a local LLM for natural answers.)"
    return text


def _notable_markers(text: str) -> str:
    hits = []
    for kw in ("TODO", "FIXME", "error", "failed", "warning", "API key", "token"):
        if kw.lower() in text.lower():
            hits.append(kw)
    return ", ".join(hits) if hits else "no obvious markers"