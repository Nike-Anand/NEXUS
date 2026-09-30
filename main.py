"""NEXUS - entry point that wires capture, OCR, context, rules and LLM together.

Run modes:
  python main.py                 GUI overlay (needs a display / Tkinter)
  python main.py --cli           headless CLI alert output (great for demos)
  python main.py --demo          feed the demo assets and show sample alerts
  python main.py --once          process a single frame (or demo) and exit

Threading model: capture/OCR/context run in a worker thread; alerts are pushed
through a queue. The UI (or CLI) lives on the main thread.
"""
from __future__ import annotations

import argparse
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from nexus.capture import ScreenCapture, active_window_title
from nexus.context import ContextEngine
from nexus.llm import LLM
from nexus.ocr import OCR, TextLine
from nexus.rules import Alert, RuleEngine
from nexus.utils import detect_onnx_providers, load_config, platform_info, setup_logging

log = setup_logging("nexus")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NEXUS on-device context-aware assistant")
    p.add_argument("--cli", action="store_true", help="headless CLI mode (no GUI)")
    p.add_argument("--ui", action="store_true", help="force GUI mode")
    p.add_argument("--demo", action="store_true", help="feed demo_assets and demo alerts")
    p.add_argument("--once", action="store_true", help="one iteration then exit")
    p.add_argument("--interval", type=float, default=None, help="capture interval seconds")
    p.add_argument("--cooldown", type=float, default=None, help="alert cooldown seconds")
    p.add_argument("--config", default=None, help="path to a custom config.yaml")
    return p


class NexusApp:
    """Owns the capture -> OCR -> context -> rules (+ LLM) pipeline."""

    def __init__(self, config: dict, demo: bool = False, providers=None,
                 primary: Optional[str] = None):
        self.config = config
        self.demo = demo
        self.providers = list(providers or [])
        self.primary = primary
        cap = config.get("capture", {})
        rules = config.get("rules", {})
        self.capture = ScreenCapture(
            interval=float(cap.get("interval", 3.0)),
            min_change=float(cap.get("min_change", 0.02)),
            monitor=int(cap.get("monitor", 0)),
        )
        self.ocr = None
        self._ocr_cfg = bool(config.get("ocr", {}).get("use_npu", True))
        self.context = ContextEngine(
            max_frames=int(config.get("context", {}).get("max_frames", 20)),
            text_budget=int(config.get("context", {}).get("text_budget", 3000)),
        )
        self.rules = RuleEngine(
            cooldown=float(rules.get("cooldown", 30.0)),
            suppress=list(rules.get("suppress", [])),
        )
        self.llm = LLM(config.get("llm", {}))
        self.alerts: "queue.Queue[Alert]" = queue.Queue()
        self.running = True
        self.paused = threading.Event()
        self.demo_lines: List[str] = self._load_demo_assets()

    # -- demo helpers --------------------------------------------------------
    @staticmethod
    def _load_demo_assets() -> List[str]:
        base = Path(__file__).resolve().parent / "demo_assets"
        text = ""
        for f in sorted(base.glob("*.txt")):
            try:
                text += f.read_text(encoding="utf-8") + "\n"
            except OSError:
                continue
        return [line.strip() for line in text.splitlines() if line.strip()]
# -- main loop -----------------------------------------------------------
    def worker_loop(self) -> None:
        """Capture / OCR / contextualize / alert, one iteration per cycle."""
        last = 0.0
        while self.running:
            while self.paused.is_set():
                if not self.running:
                    return
                time.sleep(0.2)
            if self.demo and not self.demo_lines:
                break  # demo feed consumed
            if not self.demo and self.capture.should_wait(last):
                time.sleep(0.25)
                continue
            started = time.perf_counter()
            try:
                lines = self._grab_and_ocr()
                if lines:
                    snap = self.context.update(lines, active_window_title())
                    for alert in self.rules.evaluate(snap.text):
                        self.alerts.put(alert)
            except Exception as exc:  # noqa: BLE001
                log.warning("worker cycle failed (degrading): %s", exc)
            last = time.perf_counter()
            if not self.demo:
                log.info("cycle done in %.0f ms (backend=%s)",
                         (last - started) * 1000, self.ocr.backend)

    def _grab_and_ocr(self) -> List[TextLine]:
        if self.demo:
            if self.demo_lines:
                text = self.demo_lines.pop(0)
                return [TextLine(text=text, conf=1.0, box=[])]
            return []
        if self.ocr is None:  # build lazily so demo mode never downloads models
            self.ocr = OCR(use_npu=self._ocr_cfg)
        frame = self.capture.capture()
        if frame is None:  # unchanged screen
            return []
        return self.ocr.run(frame.image)

    # -- control -------------------------------------------------------------
    def stop(self) -> None:
        self.running = False

    def pause(self) -> None:
        self.paused.set()

    def resume(self) -> None:
        self.paused.clear()

    def toggle_pause(self) -> bool:
        if self.paused.is_set():
            self.resume()
            return False
        self.pause()
        return True

    def is_paused(self) -> bool:
        return self.paused.is_set()

    # -- status for the UI ---------------------------------------------------
    def status(self) -> Dict[str, str]:
        """A snapshot of backends / mode for the status bar (honest labels)."""
        from nexus.utils import backend_status  # local import to avoid cycles

        ocr_backend = (
            self.ocr.backend
            if self.ocr is not None else self.primary or "pending"
        )
        return {
            "npu": backend_status(self.providers),
            "ocr": ocr_backend,
            "llm": self.llm.backend,
            "mode": "demo" if self.demo else "live",
            "paused": "paused" if self.paused.is_set() else "running",
        }

    # -- CLI output ----------------------------------------------------------
    def drain_alerts(self) -> None:
        try:
            while True:
                alert = self.alerts.get_nowait()
                print_cli_alert(alert)
        except queue.Empty:
            pass


def print_cli_alert(alert: Alert) -> None:
    flag = {"critical": "[!]", "high": "[!]", "medium": "[~]", "low": "[i]"}[
        alert.severity
    ]
    print(f"{flag} {alert.severity.upper():8} {alert.category:16} {alert.matched}")
    print(f"      -> {alert.suggestion}")


def run_cli(app: NexusApp, once: bool) -> None:
    """Run the worker, printing alerts to the terminal."""
    thread = threading.Thread(target=app.worker_loop, daemon=True)
    thread.start()
    try:
        while True:
            app.drain_alerts()
            if once and app.demo and not app.demo_lines and not thread.is_alive():
                break  # demo feed consumed and the worker finished
            if once and not app.demo:
                # live --once: give the worker time for a full capture+OCR cycle
                time.sleep(2.5)
                break
            if not thread.is_alive():
                break  # live worker exited unexpectedly
            time.sleep(0.2)
    except KeyboardInterrupt:
        log.info("Stopping NEXUS.")
    finally:
        app.drain_alerts()  # catch any alerts still queued
        app.stop()


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.interval is not None:
        config.setdefault("capture", {})["interval"] = args.interval
    if args.cooldown is not None:
        config.setdefault("rules", {})["cooldown"] = args.cooldown

    info = platform_info()
    providers, primary = detect_onnx_providers()
    print(f"NEXUS starting | {info}")
    print(f"ONNX providers: {providers} | primary={primary}")
    log.info("Platform: %s", info)

    app = NexusApp(config, demo=args.demo, providers=providers, primary=primary)
    if args.cli:
        run_cli(app, once=args.once)
        return 0
    from nexus.ui import NexusUI

    NexusUI(app, providers=providers, primary=primary).run(once=args.once)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())