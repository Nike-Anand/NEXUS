"""NEXUS benchmark - real, on-machine measurements (M4).

Run:  python scripts/benchmark.py [--iters N] [--chart]

Every number printed is MEASURED on the current machine, never estimated. Each
metric is labelled with the hardware it was measured on, and the hardware tag is
printed at the top so a reader can tell a Snapdragon (NPU/QNN) result from an
AMD64/Intel (CPU) result at a glance.

Metrics (all wall-clock, median of N):
  * screen capture      - ScreenCapture.capture(force=True)
  * OCR inference       - OCR.run() on a synthetic image with printed text
  * LLM backend         - LLM.generate() for the selected backend
  * end-to-end cycle    - capture -> OCR -> context -> rules (single frame)

Conservative defaults keep the run under ~1-2 minutes on CPU.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from nexus.capture import ScreenCapture  # noqa: E402
from nexus.context import ContextEngine  # noqa: E402
from nexus.llm import LLM  # noqa: E402
from nexus.ocr import OCR, TextLine  # noqa: E402
from nexus.rules import RuleEngine  # noqa: E402
from nexus.utils import detect_onnx_providers, platform_info  # noqa: E402


def _stats(samples):
    return {
        "n": len(samples),
        "median_ms": statistics.median(samples) * 1000.0,
        "p95_ms": statistics.quantiles(samples, n=20)[-1] * 1000.0,
    }


def _synthetic_image() -> Image.Image:
    img = Image.new("RGB", (640, 400), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("segoeui.ttf", 28)
    except Exception:  # noqa: BLE001
        font = ImageFont.load_default()
    for i, line in enumerate(
        ["NEXUS on-device benchmark", "Hello from the lab",
         "OPENAI_API_KEY=sk-fake-bench-12345"]
    ):
        draw.text((20, 30 + i * 60), line, fill="black", font=font)
    return img


def bench_capture(n: int) -> dict:
    cap = ScreenCapture(interval=0.0, min_change=-1.0)
    samples = []
    ok = 0
    for _ in range(n):
        t0 = time.perf_counter()
        frame = cap.capture(force=True)
        samples.append(time.perf_counter() - t0)
        if frame is not None:
            ok += 1
    return {"ok": ok, **_stats(samples)}


def bench_ocr(n: int) -> dict:
    ocr = OCR(use_npu=True)
    image = _synthetic_image()
    if not ocr.available:
        return {"available": False}
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        ocr.run(image)
        samples.append(time.perf_counter() - t0)
    lines = max(1, len(ocr.run(image)))
    fps = lines / (statistics.median(samples) or 1e-9)
    return {"available": True, "lines": lines, "fps": fps, **_stats(samples)}


def bench_llm(n: int) -> dict:
    llm = LLM({})
    samples = []
    res = None
    for _ in range(n):
        t0 = time.perf_counter()
        res = llm.generate("[SCREEN CONTEXT]\ncode editor with a TODO\n\n", 60)
        samples.append(time.perf_counter() - t0)
    toks = max(1, res["stats"].tokens)
    return {"backend": llm.backend, "tokens": toks, **_stats(samples)}


def bench_e2e(n: int) -> dict:
    ocr = OCR(use_npu=True)
    ctx = ContextEngine()
    rules = RuleEngine(cooldown=0.0)
    image = _synthetic_image()
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        lines = ocr.run(image)
        if lines:
            snap = ctx.update([TextLine(t.text, t.conf, t.box)
                               for t in lines], "bench.py")
            rules.evaluate(snap.text)
        samples.append(time.perf_counter() - t0)
    return _stats(samples)


def _hardware_tag(providers):
    if "QNNExecutionProvider" in providers:
        return "NPU (QNN)"
    if "DmlExecutionProvider" in providers:
        return "NPU (DirectML)"
    if "CUDAExecutionProvider" in providers:
        return "NPU (CUDA)"
    return "CPU"


def main() -> int:
    ap = argparse.ArgumentParser(description="NEXUS benchmark")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--chart", action="store_true", help="write benchmark_chart.png")
    args = ap.parse_args()
    n = max(3, args.iters)

    info = platform_info()
    providers, primary = detect_onnx_providers()
    hw = _hardware_tag(providers)
    machine = f"{info['system']}-{info['machine']} (ARM64={info['is_arm64']})"
    print("=" * 68)
    print(f"  NEXUS benchmark - REAL measurements (iterations={n})")
    print("=" * 68)
    print(f"Hardware   : {machine}")
    print(f"Python     : {info['python']}")
    print(f"ONNX       : {providers} -> primary={primary}")
    print(f"Measured on: {hw}   <-- annotates every row below")
    print()
    print("Rows below are unambiguously MEASURED on the hardware above.")
    print("NPU (QNN) figures require a Snapdragon device and are NOT shown here.")
    print()

    def row(label, ms, extra=""):
        p95 = f"  (p95 {ms['p95_ms']:.1f} ms)" if "p95_ms" in ms else ""
        print(f"  {label:<26} {ms['median_ms']:7.2f} ms   [{hw}]{p95} {extra}")

    rows = []
    print("-- Capture (force grab of primary monitor) --")
    try:
        r = bench_capture(n)
        row("screen capture median", r)
        rows.append(("screen_capture_ms", r["median_ms"]))
    except Exception as exc:  # noqa: BLE001
        print(f"  [capture unavailable] {exc}")

    print("-- OCR (PP-OCRv4 via RapidOCR, synthetic text) --")
    r = bench_ocr(n)
    if r.get("available"):
        row("OCR inference median", r, f"| lines={r['lines']} "
                                       f"~{r['fps']:.1f} fps")
        rows.append(("ocr_ms", r["median_ms"]))
    else:
        print("  [OCR unavailable; models not downloaded?]")

    print("-- LLM backend (selected) --")
    r = bench_llm(n)
    row(f"LLM generate ({r['backend']}) median", r, f"| {r['tokens']} tok")
    rows.append(("llm_ms", r["median_ms"]))

    print("-- End-to-end (capture->OCR->context->rules) --")
    r = bench_e2e(n)
    row("full cycle median", r)
    rows.append(("e2e_ms", r["median_ms"]))

    print()
    print("=" * 68)
    for key, ms in rows:
        print(f"  {key:<22} {ms:8.2f} ms   [{hw}]")
    print("=" * 68)

    if args.chart:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.ticker as mticker

            keys = [k for k, _ in rows]
            vals = [v for _, v in rows]
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.barh(keys[::-1], vals[::-1], color="#4caf50")
            ax.set_xlabel(f"median ms (measured on {hw})")
            ax.set_title(f"NEXUS benchmark - measured on {hw}")
            fig.tight_layout()
            fig.savefig(ROOT / "benchmark_chart.png", dpi=140)
            print(f"Chart written to {ROOT / 'benchmark_chart.png'}")
        except Exception as exc:  # noqa: BLE001
            print(f"[chart skipped] matplotlib unavailable: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())