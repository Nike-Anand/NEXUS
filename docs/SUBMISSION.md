# NEXUS — Submission notes (M1–M5)

## What this is
NEXUS is an on-device, context-aware screen assistant for Snapdragon (Windows on
ARM64) laptops, built for the M1–M5 milestone. It captures the screen, OCRs it
locally (PP-OCRv4 via RapidOCR on ONNX Runtime), builds a rolling context, runs a
deterministic rules engine for alerts, and offers an LLM chain (with a pure
template fallback) for Explain/Assist.

## Scope covered
| Milestone | Delivered |
|-----------|-----------|
| **M1** | Capture → OCR → context → rules → alerts pipeline, CLI + lazy OCR, all-5 critical bugs fixed. 24 unit tests. |
| **M2** | Tkinter overlay: severity-colored toasts (dedup/cooldown), control panel with pause/resume, honest status bar (`NPU (QNN)` only when confirmed), "100% on-device" badge. Works in `--demo` and live. |
| **M3** | LLM chain (onnxruntime-genai → Ollama → llama-cpp → template) with graceful fallback; streaming **Explain** (Ctrl+Alt+E) and **Assist** (Ctrl+Alt+A) panels; works fully in template/no-LLM mode. |
| **M4** | `scripts/benchmark.py` — real measurements only, each labelled with the hardware it was measured on. |
| **M5** | README (Mermaid diagram), `docs/DEMO_SCRIPT.md`, this file. |

> **Act mode is intentionally NOT built** (M6 roadmap only). It is documented in
> the README and left `act.enabled: false`.

---

## Measured performance (M4) — real, on this hardware
**Hardware:** Windows 11 · `AMD64` · Python 3.11.9 · onnxruntime providers
`['AzureExecutionProvider', 'CPUExecutionProvider']` → **CPU path (no QNN)**.

| Metric | Median (10 iters) | Hardware |
|--------|-------------------|----------|
| Screen capture (force grab) | 28.34 ms | CPU (AMD64 dev VM) |
| OCR inference (PP-OCRv4, 3 lines) | 5298.80 ms (~0.6 fps) | CPU (AMD64 dev VM) |
| LLM generate (template backend) | 0.02 ms | CPU (AMD64 dev VM) |
| End-to-end cycle (capture→OCR→context→rules) | 5227.07 ms | CPU (AMD64 dev VM) |

These are **measured**, not estimated. Rerun any time:
```powershell
.venv\Scripts\python.exe scripts/benchmark.py --iters 10
```

> **Not measured here:** NPU (QNN) figures. They require a Snapdragon device with
> `onnxruntime-qnn` installed; on the AMD64 dev machine there is no NPU to measure,
> so NEXUS honestly reports CPU and never fabricates an NPU number.

---

## How to reproduce
```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/check_env.py              # PASS
python -m pytest tests/ -q               # 32 passed
python main.py --demo --cli --once       # all 10 demo alerts
python main.py --demo --ui --once        # GUI toasts + controls, auto-closes
python scripts/benchmark.py --iters 10   # real measurements
```

---

## Verification summary
- `check_env.py` → **PASS** (all required deps; template LLM backend selected).
- `pytest tests/` → **32 passing** (`test_rules.py` + new headless `test_ui.py`).
- `--demo --cli --once` → all **10** demo alerts printed (pre-verified in M1).
- `--demo --ui --once` → overlay boots, toasts render, auto-closes cleanly.
- Explain/Assist panels **stream** template output in demo mode (verified).

---

## Known limitations (honest)
1. **NPU untestable here** — staged end-to-end, but real QNN numbers need a
   Snapdragon device. CPU path verified.
2. **OCR is the bottleneck on CPU** (~5 s/inference). Expected to improve
   substantially on QNN; not a regression and not hidden.
3. **System-wide hotkeys optional** — require the `keyboard` package. Without it,
   hotkeys work while the control panel has focus; buttons always work.
4. **`opencv-python` on ARM64** runs under x64 emulation (no native wheel yet).
5. Em-dash / `·` glyphs can look like `ù` in PowerShell console output (code page)
   even though they render correctly in the GUI.

## Deliverables in this milestone
- `nexus/ui.py` (new) — Tkinter overlay, toasts, control panel, streaming panels.
- `nexus/llm.py`, `nexus/explain.py`, `nexus/assist.py` — LLM chain + prompts
  (M3 streaming support surfaced in the UI).
- `nexus/utils.py`, `nexus/rules.py` — honest `backend_status`, pause/status API.
- `main.py` — GUI/CLI dispatch, pause/resume + status on `NexusApp`.
- `scripts/benchmark.py` (new) — real measurements.
- `tests/test_ui.py` (new), `config.yaml`, `requirements.txt`, `README.md`,
  `docs/DEMO_SCRIPT.md`, `docs/SUBMISSION.md`.