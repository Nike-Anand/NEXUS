"""Screen capture with intelligent change detection.

Uses ``mss`` to grab the active monitor as a PIL image every N seconds. If the
perceptual hash of the frame has not moved beyond a threshold, the frame is
skipped so you do not waste compute (and battery) on an idle display.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

from PIL import Image

from .utils import hamming_distance, image_dhash, setup_logging

log = setup_logging(__name__)


@dataclass
class Frame:
    """One captured frame plus metadata."""

    image: Image.Image
    monitor: dict
    captured_at: datetime
    changed: bool
    dhash: str


class ScreenCapture:
    """Grab the active (primary/most relevant) monitor periodically."""

    def __init__(self, interval: float = 3.0, min_change: float = 0.02, monitor: int = 0):
        self.interval = interval
        self.min_change = min_change  # fraction of bits that must differ
        self.monitor_index = monitor
        self._last_dhash: Optional[str] = None
        self._last_ts = 0.0

    # -- helpers --------------------------------------------------------------
    def _grab_monitor(self):
        """Lazily import mss and grab the target monitor as a PIL image."""
        import mss

        with mss.mss() as sct:
            monitors = sct.monitors
            idx = min(self.monitor_index, len(monitors) - 1)
            mon = monitors[idx]
            raw = sct.grab(mon)
            img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
            return mon, img

    # -- public API -----------------------------------------------------------
    def capture(self, force: bool = False) -> Optional[Frame]:
        """Capture one frame, returning None if the screen has not changed."""
        try:
            mon, img = self._grab_monitor()
        except Exception as exc:  # pragma: no cover - hardware dependant
            log.warning("Screen capture failed: %s", exc)
            return None

        dhash = image_dhash(img)
        changed = True
        if self._last_dhash is not None:
            distance = hamming_distance(self._last_dhash, dhash)
            ratio = distance / max(1, len(dhash))
            changed = ratio >= self.min_change or force

        self._last_dhash = dhash
        frame = Frame(
            image=img,
            monitor=mon,
            captured_at=datetime.now(),
            changed=changed or self._last_ts == 0.0,
            dhash=dhash,
        )
        self._last_ts = time.time()
        return frame if (changed or force) else None

    def should_wait(self, last_run: float) -> bool:
        """True while we should sleep before the next capture."""
        return (time.time() - last_run) < self.interval


def active_window_title() -> str:
    """Best-effort active window title via pygetwindow, with a safe fallback."""
    try:
        import pygetwindow as gw

        win = gw.getActiveWindow()
        return win.title if win and getattr(win, "title", None) else ""
    except Exception:
        return ""