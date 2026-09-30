"""Unit tests for the M2/M3 UI and LLM-chain helpers.

These tests avoid opening real Tk windows so they are safe to run headless.
Run with:  python -m pytest tests/test_ui.py -v
"""
from nexus.assist import build_assist_prompt
from nexus.context import ContextSnapshot
from nexus.explain import build_explain_prompt
from nexus.llm import LLM, _template_answer
from nexus.ui import severity_color
from nexus.utils import backend_status

# --- severity colors ---------------------------------------------------------
def test_severity_color_known_levels():
    assert severity_color("critical")[0] == "#b71c1c"  # red
    assert severity_color("low")[0] == "#2e7d32"        # green


def test_severity_color_unknown_falls_back():
    assert severity_color("nope")[0] == "#455a64"


# --- honest backend reporting -------------------------------------------------
def test_backend_status_npu_only_when_qnn_confirmed():
    assert backend_status(["QNNExecutionProvider", "CPUExecutionProvider"]) == "NPU (QNN)"
    assert backend_status(["DmlExecutionProvider", "CPUExecutionProvider"]) == "NPU (DirectML)"
    assert backend_status(["CUDAExecutionProvider"]) == "NPU (CUDA)"


def test_backend_status_never_fabricates_npu():
    # An AMD64 box that only exposes CPU must report CPU, never NPU.
    assert backend_status(["AzureExecutionProvider", "CPUExecutionProvider"]) == "CPU"
    assert backend_status([]) == "unavailable"


# --- LLM streaming / template fallback ------------------------------------------
def test_template_answer_never_raises_and_is_grounded():
    text = _template_answer("[SCREEN CONTEXT]\nOPENAI_API_KEY=sk-deploy-code-here\n\n")
    assert "screen" in text or "deploy" in text
    assert "Template mode" in text


def test_llm_stream_reassembles_template_text():
    llm = LLM({"backend_order": ["onnx", "ollama", "llama_cpp"]})
    assert llm.backend == "template"  # none installed in this env
    chunks = []

    def on_token(chunk):
        chunks.append(chunk)

    out = llm.stream("[SCREEN CONTEXT]\ncode editor with a TODO\n\n",
                     on_token, max_tokens=60)
    assert "".join(chunks) == out
    assert len(out) > 0


# --- prompt grounding -----------------------------------------------------------
def test_explain_prompt_contains_screen_context():
    snap = ContextSnapshot(text="rust parser failing on empty input",
                           app_type="editor", active_window="main.rs")
    prompt = build_explain_prompt(snap)
    assert "rust parser" in prompt
    assert "main.rs" in prompt


def test_assist_prompt_contains_request_and_context():
    snap = ContextSnapshot(text="deploy migration to prod",
                           app_type="terminal", active_window="pwsh")
    prompt = build_assist_prompt(snap, request="should I roll back?")
    assert "roll back" in prompt
    assert "deploy migration" in prompt