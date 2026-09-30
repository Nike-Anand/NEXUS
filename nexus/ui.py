"""Tkinter overlay for NEXUS: alert toasts + a small control panel (M2/M3).

The capture/OCR/rules pipeline runs in a worker thread (see ``main.NexusApp``).
Everything in this module only renders on the Tk main thread:

  * severity-coloured toasts, auto-dismissing and stacked top-right. Dedup /
    cooldown lives in the rules engine, so the overlay only draws NEW alerts.
  * a small always-on-top control window with Pause/Resume, an honest status bar
    (shows ``NPU (QNN)`` only when QNN was actually registered at runtime) and a
    "100% on-device" badge.
  * Explain / Assist hotkeys and buttons that stream LLM output into a panel.

Tkinter is a stdlib dependency (no PySide6), so this stays clean on Windows
ARM64. Importing this module never creates a window - that only happens when
``NexusUI(...).run()`` is called on the main thread.
"""
from __future__ import annotations

import logging
import queue
import threading
import tkinter as tk
from tkinter import ttk
from typing import List, Optional

from . import assist as assist_mod
from . import explain as explain_mod
from .utils import setup_logging

log = setup_logging(__name__)

# (background, foreground) per severity - high risk = red, etc.
SEVERITY_COLORS = {
    "critical": ("#b71c1c", "#ffffff"),
    "high": ("#e65100", "#ffffff"),
    "medium": ("#f9a825", "#1a1a1a"),
    "low": ("#2e7d32", "#ffffff"),
}
_PANEL_BG = "#1e2430"
_PANEL_FG = "#e8eaed"
_ACCENT = "#4caf50"


def severity_color(severity: str) -> tuple:
    """Colour pair ``(background, foreground)`` for a severity level."""
    return SEVERITY_COLORS.get(severity or "low", ("#455a64", "#ffffff"))


class NexusUI:
    """Owns the Tk root, toasts, control panel and the streaming result panels."""

    def __init__(self, app, providers=None, primary: Optional[str] = None):
        self.app = app
        self.providers = list(providers or app.providers)
        self.primary = primary or app.primary
        self._toasts: List[tk.Toplevel] = []
        self._global_hotkeys = False
        self._worker = None

        self._root = tk.Tk()
        self._root.title("NEXUS — 100% on-device")
        self._root.configure(bg=_PANEL_BG)
        self._root.resizable(False, False)
        self._root.attributes("-topmost", True)

        self._build_panel()
        self._register_hotkeys()
        self._refresh_status()

    # ------------------------------------------------------------------ panel
    def _label(self, parent, text, **kw) -> tk.Label:
        defaults = dict(bg=_PANEL_BG, fg=_PANEL_FG, font=("Segoe UI", 9))
        defaults.update(kw)
        return tk.Label(parent, text=text, **defaults)

    def _build_panel(self) -> None:
        root = self._root
        f = tk.Frame(root, bg=_PANEL_BG, padx=14, pady=10)
        f.pack(fill="both", expand=True)

        self._label(f, "NEXUS", font=("Segoe UI", 14, "bold"),
                    fg=_ACCENT).grid(row=0, column=0, columnspan=3, sticky="w")
        self._label(f, "100% on-device", font=("Segoe UI", 8),
                    fg="#8bc34a").grid(row=1, column=0, columnspan=3, sticky="w")

        status = tk.LabelFrame(f, text="Status", bg=_PANEL_BG, fg=_PANEL_FG,
                               font=("Segoe UI", 8, "bold"), bd=0)
        status.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(8, 4))

        self._mode_lbl = self._label(status, "mode")
        self._mode_lbl.grid(row=0, column=0, sticky="w", padx=8, pady=2)
        self._npu_lbl = self._label(status, "accelerator")
        self._npu_lbl.grid(row=1, column=0, sticky="w", padx=8, pady=2)
        self._llm_lbl = self._label(status, "llm")
        self._llm_lbl.grid(row=2, column=0, sticky="w", padx=8, pady=2)
        self._paused_lbl = self._label(status, "running")
        self._paused_lbl.grid(row=3, column=0, sticky="w", padx=8, pady=2)

        btns = tk.Frame(f, bg=_PANEL_BG)
        btns.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(6, 4))
        self._pause_btn = ttk.Button(btns, text="Pause watch",
                                     command=self._toggle_pause)
        self._pause_btn.pack(side="left", padx=(0, 6))
        ttk.Button(btns, text="Explain  (Ctrl+Alt+E)",
                   command=self._start_explain).pack(side="left", padx=6)
        ttk.Button(btns, text="Assist  (Ctrl+Alt+A)",
                   command=self._start_assist).pack(side="left", padx=6)

        hint = self._label(f, "Hotkeys work when this panel has focus.\n"
                              "Install 'keyboard' for system-wide hotkeys.",
                           font=("Segoe UI", 7), fg="#9aa5b1")
        hint.grid(row=4, column=0, columnspan=3, sticky="w", pady=(6, 0))

    # ---------------------------------------------------------------- pause
    def _toggle_pause(self) -> None:
        self.app.toggle_pause()
        self._refresh_status()

    # ---------------------------------------------------------------- status
    def _refresh_status(self) -> None:
        st = self.app.status()
        self._mode_lbl.config(text=f"Mode: {st['mode']}")
        self._npu_lbl.config(text=f"Detector: {st['ocr']}  ({st['npu']})")
        self._llm_lbl.config(text=f"LLM backend: {st['llm']}")
        running = st["paused"] == "running"
        self._paused_lbl.config(text="State: running" if running else "State: paused",
                                fg=_ACCENT if running else "#f9a825")
        self._pause_btn.config(text="Pause watch" if running else "Resume watch")

    # ---------------------------------------------------------------- toasts
    def _show_toast(self, alert) -> None:
        cfg = self.app.config.get("ui", {})
        max_toasts = int(cfg.get("max_toasts", 3))
        timeout_ms = int(cfg.get("toast_timeout_ms", 5000))
        while len(self._toasts) >= max_toasts:  # keep the newest buffered
            old = self._toasts.pop(0)
            try:
                old.destroy()
            except Exception:  # noqa: BLE001
                pass

        bg, fg = severity_color(alert.severity)
        top = tk.Toplevel(self._root)
        top.overrideredirect(True)
        top.attributes("-topmost", True)
        frame = tk.Frame(top, bg=bg, padx=12, pady=8)
        frame.pack(fill="both", expand=True)
        tk.Label(frame, text=f"{alert.severity.upper()} · "
                             f"{alert.category.replace('_', ' ')}",
                 bg=bg, fg=fg, font=("Segoe UI", 8, "bold")).pack(anchor="w")
        tk.Label(frame, text=alert.matched, bg=bg, fg=fg, wraplength=360,
                 justify="left", font=("Segoe UI", 9)).pack(anchor="w")

        self._toasts.append(top)
        self._stack_toasts()

        def dismiss():
            if top in self._toasts:
                self._toasts.remove(top)
            try:
                top.destroy()
            except Exception:  # noqa: BLE001
                pass
            self._stack_toasts()

        top.after(timeout_ms, dismiss)

    def _stack_toasts(self) -> None:
        y = 16
        for top in self._toasts:
            try:
                top.update_idletasks()
                w = top.winfo_reqwidth()
                h = top.winfo_reqheight()
                top.geometry(f"+{top.winfo_screenwidth() - w - 16}+{y}")
                y += h + 10
            except Exception:  # noqa: BLE001
                continue

    # ----------------------------------------------------------- result panel
    def _open_result(self, title: str):
        win = tk.Toplevel(self._root)
        win.title(title)
        win.geometry("540x360")
        win.attributes("-topmost", True)
        win.configure(bg=_PANEL_BG)
        txt = tk.Text(win, wrap="word", font=("Consolas", 10),
                      bg="#0f1117", fg="#d6dce4", insertbackground=_PANEL_FG,
                      padx=10, pady=10, relief="flat")
        sb = ttk.Scrollbar(win, command=txt.yview)
        txt.configure(yscrollcommand=sb.set)
        txt.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        return win, txt

    # ------------------------------------------------------------ LLM stream
    def _stream_error(self, kind: str, message: str) -> None:
        win, txt = self._open_result(f"NEXUS — {kind.title()}")
        txt.insert("1.0", message)

    def _stream_generate(self, kind: str) -> None:
        snap = self.app.context.current()
        if not snap.text and not snap.active_window:
            self._stream_error(kind, "No screen context yet. Capture a frame "
                                     "first (or run in demo mode).")
            return

        title = "NEXUS — Explain" if kind == "explain" else "NEXUS — Assist"
        _, txt = self._open_result(title)
        llm_cfg = self.app.config.get("llm", {})
        if kind == "explain":
            prompt = explain_mod.build_explain_prompt(snap)
            max_tokens = int(llm_cfg.get("explain_max_tokens", 180))
        else:
            prompt = assist_mod.build_assist_prompt(snap)
            max_tokens = int(llm_cfg.get("assist_max_tokens", 200))

        q: queue.Queue = queue.Queue()

        def runner() -> None:
            try:
                self.app.llm.stream(prompt, lambda chunk: q.put(chunk),
                                    max_tokens=max_tokens)
            except Exception as exc:  # noqa: BLE001
                q.put(f"\n[LLM error, degraded to template] {exc}")
            finally:
                q.put(None)

        threading.Thread(target=runner, daemon=True).start()
        txt.insert("1.0", "\u2026")
        self._poll_stream(txt, q)

    def _poll_stream(self, txt: tk.Text, q: queue.Queue) -> None:
        try:
            while True:
                chunk = q.get_nowait()
                if chunk is None:
                    # stream done: drop the leading "…" placeholder only
                    if txt.winfo_exists():
                        try:
                            txt.delete("1.0", "1.1")
                        except Exception:  # noqa: BLE001
                            pass
                    return
                if txt.winfo_exists():
                    # overwrite the placeholder with the first real token
                    if txt.get("1.0", "end-1c") == "\u2026":
                        txt.delete("1.0", "1.1")
                    txt.insert("end", chunk)
                    txt.see("end")
        except queue.Empty:
            pass
        except Exception:  # noqa: BLE001 - window closed mid-stream
            return
        try:
            if txt.winfo_exists():
                self._root.after(60, lambda: self._poll_stream(txt, q))
        except Exception:  # noqa: BLE001
            return

    def _start_explain(self) -> None:
        self._stream_generate("explain")

    def _start_assist(self) -> None:
        self._stream_generate("assist")

    # ---------------------------------------------------------------- hotkeys
    def _register_hotkeys(self) -> None:
        # In-focus bindings always work without extra dependencies.
        try:
            self._root.bind_all("<Control-Alt-e>", lambda e: self._start_explain())
            self._root.bind_all("<Control-Alt-a>", lambda e: self._start_assist())
        except Exception:  # noqa: BLE001
            pass
        # System-wide hotkeys when the optional 'keyboard' package is present.
        try:
            import keyboard  # type: ignore

            keyboard.add_hotkey("ctrl+alt+e", self._start_explain)
            keyboard.add_hotkey("ctrl+alt+a", self._start_assist)
            self._global_hotkeys = True
            log.info("System-wide hotkeys registered (keyboard package).")
        except Exception:  # noqa: BLE001
            self._global_hotkeys = False

    # --------------------------------------------------------------- polling
    def _poll(self) -> None:
        try:
            while True:
                self._show_toast(self.app.alerts.get_nowait())
        except queue.Empty:
            pass
        self._refresh_status()
        self._root.after(150, self._poll)

    # ------------------------------------------------------------------- run
    def run(self, once: bool = False) -> None:
        """Start the worker thread and enter the Tk event loop."""
        self._worker = threading.Thread(target=self.app.worker_loop, daemon=True)
        self._worker.start()
        if once:
            # convenience for demos / automated checks: auto-close shortly after
            self._root.after(3500, self._root.destroy)
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._root.after(200, self._poll)
        try:
            self._root.mainloop()
        finally:
            self.app.stop()
            self._stop_global_hotkeys()

    def _on_close(self) -> None:
        self.app.stop()
        self._root.destroy()

    def _stop_global_hotkeys(self) -> None:
        if self._global_hotkeys:
            try:
                import keyboard  # type: ignore

                keyboard.unhook_all()
            except Exception:  # noqa: BLE001
                pass