# NEXUS demo script (for a video / live walkthrough)

> All sample keys and numbers in `demo_assets\` are **FAKE** (see that folder's
> README). Nothing in this demo touches a real credential or real personal data.

Time budget: **~4 minutes**. These commands were verified on a Windows 11 VM
(AMD64, CPU path). On a Snapdragon device the same commands work and the OCR
status bar should read `NPU (QNN)` instead of `CPU`.

---

## 0) Pre-flight (1 min)
```powershell
.venv\Scripts\activate
python scripts/check_env.py      # PASS, note the ONNX providers + LLM backend
python -m pytest tests/ -q        # 32 passed
```

**Narrate:** "Everything on screen is parsed locally; here's the env self-check."

---

## 1) Headless demo — show all 10 alerts (30 s)
```powershell
python main.py --demo --cli --once
```
**Expected output:** one alert per demo line — the OpenAI key (`[!] critical`),
the AWS key, a high-entropy `client_secret`, `rm -rf`, `DROP TABLE`, `git push
--force`, `DELETE FROM users`, `curl | sudo sh`, a fake Luhn card and a fake
Aadhaar. Alerts are masked (e.g. `sk-fak...`).

**Narrate:** "That's the deterministic rule engine — no LLM needed, all local."

---

## 2) GUI demo — toasts + control panel (1 min)
```powershell
python main.py --demo --ui --once
```
**Shows:**
- The control panel: **NEXUS · 100% on-device** badge, status bar
  (`Detector: CPU`, `LLM backend: template`, `State: running`).
- Severity-colored **toasts** roll in top-right (red = critical/high, amber =
  medium, green = low), auto-dismiss and stack.
- Click **Pause watch** → state flips to *paused*; **Resume watch** → back.
- (Interactive) click **Explain** / **Assist** → a panel *streams* the template's
  answer about the demo "editor" screen.

**Narrate:** "Pause/resume, honest accelerator label, and the streaming Explain
panel — which degrades gracefully to a fully on-device template when no LLM is
installed."

---

## 3) Live capture → OCR → rules (1 min, optional on real screen)
```powershell
python main.py --cli --once
```
Opens a live capture, runs one full OCR cycle, and (if your screen happens to show
a sample key/risk) prints the matching alert. If your screen has no risk text, the
cycle simply completes — that's expected.

**Narrate:** "The exact same pipeline that drives the GUI, headless."

---

## 4) Explain / Assist in action (30 s, interactive)
While the GUI demo is running and a frame has been captured:
- Press **Ctrl+Alt+E** (or click **Explain**) — "what am I working on?" streams in.
- Press **Ctrl+Alt+A** (or click **Assist**) — next steps stream in.

> Hotkeys fire while the control panel has focus. For **system-wide** hotkeys,
> install the optional `keyboard` package (see README).

---

## 5) Performance (optional, 30 s)
```powershell
python scripts/benchmark.py --iters 5
```
Prints real, hardware-labelled medians (capture/OCR/LLM/e2e). On this AMD64 dev VM
OCR is the slowest stage (~5 s); on a Snapdragon with `onnxruntime-qnn` this is
the number that should drop dramatically.

---

## Talking points
- **Private:** capture → OCR → context → rules all run on-device; no network calls.
- **Honest:** the status bar never claims NPU unless QNN is confirmed; benchmark
  rows are labelled with the exact hardware they were measured on.
- **Resilient:** works fully in template / no-LLM mode; every optional backend
  degrades gracefully.
- **Safe by design:** rules only alert; nothing ever runs commands (Act is roadmap).