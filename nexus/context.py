"""Context Engine: rolling window of screen text + active app classification.

Keeps the last ``N`` OCR frames, the active window title, and a short
"what is the user doing" summary. App type is detected with cheap heuristics so
Explain/Assist can tailor their tone.
"""
from __future__ import annotations

import logging
import re
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, List, Optional

from .ocr import TextLine, summarize
from .utils import setup_logging

log = setup_logging(__name__)

# Heuristics to classify the foreground app type from window title / text.
_EDITOR_HINTS = (".py", ".js", ".ts", ".cpp", ".rs", ".go", ".java", "code", "vim",
                 "neovim", "pycharm", "intellij", "sublime", "notepad", "- *")
_TERM_HINTS = ("cmd", "powershell", "terminal", "git bash", "wsl", "bash")
_BROWSER_HINTS = ("chrome", "edge", "firefox", "brave", "browser", "- google")
_DOC_HINTS = ("word", "pages", "docs", "pdf", "acrobat", "powerpoint", "overleaf")


@dataclass
class AppType:
    kind: str
    confidence: float


@dataclass
class ContextSnapshot:
    """Immutable summary of what the user is currently working on."""

    text: str = ""
    app_type: str = "unknown"
    active_window: str = ""
    captured_at: Optional[datetime] = None
    last_error: str = ""


class ContextEngine:
    """Maintains a rolling window of recent screen text and window metadata."""

    def __init__(self, max_frames: int = 20, text_budget: int = 3000):
        self.max_frames = max_frames
        self.text_budget = text_budget
        self.window: Deque[ContextSnapshot] = deque(maxlen=max_frames)
        self.active_window = ""
        self.last_ocr: List[TextLine] = []

    # -- helpers --------------------------------------------------------------
    def classify_app(self, window_title: str, screen_text: str = "") -> AppType:
        """Best-effort detection of the foreground app category."""
        hay = (window_title + " " + screen_text).lower()
        for name, hints in (
            ("editor", _EDITOR_HINTS),
            ("terminal", _TERM_HINTS),
            ("browser", _BROWSER_HINTS),
            ("docs", _DOC_HINTS),
        ):
            hits = sum(1 for h in hints if h in hay)
            if hits:
                return AppType(kind=name, confidence=min(0.95, 0.5 + hits * 0.15))
        return AppType(kind="unknown", confidence=0.3)

    @staticmethod
    def _dedupe(lines: List[TextLine]) -> List[TextLine]:
        """Drop repeated lines within a single OCR pass (scrolling artefacts)."""
        seen = set()
        out = []
        for line in lines:
            key = line.text.strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(line)
        return out

    # -- public API -----------------------------------------------------------
    def update(self, lines: List[TextLine], window_title: str = "") -> ContextSnapshot:
        """Push a new OCR pass into the window and return the latest snapshot."""
        if window_title:
            self.active_window = window_title
        self.last_ocr = self._dedupe(lines)
        text = summarize(self.last_ocr, limit=self.text_budget)
        app = self.classify_app(window_title or self.active_window, text)
        snap = ContextSnapshot(
            text=text,
            app_type=app.kind,
            active_window=window_title or self.active_window,
            captured_at=datetime.now(),
        )
        self.window.append(snap)
        return snap

    def current(self) -> ContextSnapshot:
        """The most recent snapshot, or an empty one if nothing captured yet."""
        return self.window[-1] if self.window else ContextSnapshot()

    def rolling_text(self, limit: Optional[int] = None) -> str:
        """Concatenated text across the rolling window (bounded)."""
        budget = limit or self.text_budget * 2
        parts: List[str] = []
        total = 0
        for snap in self.window:
            if snap.text and total < budget:
                parts.append(snap.text)
                total += len(snap.text)
        return "\n".join(parts)


# Convenience regex used by Explain/Assist template mode.
CODE_FILE_RE = re.compile(r"[\w./\\-]+\.(py|js|ts|rs|go|cpp|c|java|sh|ps1|yaml|yml|json|txt)\b")