"""Explain: "what is the user working on?" via the local LLM.

Builds a grounded prompt from the current screen context, keeps the answer to
3-5 simple sentences, and works in streaming or single-shot mode. When no LLM
backend is present the template summarizer in ``llm`` answers instead.
"""
from __future__ import annotations

from typing import Callable, Dict

from .context import ContextSnapshot
from .llm import LLM

SYSTEM = (
    "You are NEXUS, a private on-device screen assistant. Explain what the "
    "user is working on in 3 to 5 simple, honest sentences. Mention anything "
    "notable (code, errors, TODOs, risky text). Never invent details."
)


def build_explain_prompt(snap: ContextSnapshot) -> str:
    """Assemble the Explain prompt from a context snapshot."""
    app = snap.app_type or "unknown"
    win = snap.active_window or "unknown"
    screen = (snap.text or "".strip())[:1500]  # bound context length
    return (
        f"{SYSTEM}\n\n"
        f"App type: {app}\nActive window: {win}\n\n"
        f"[SCREEN CONTEXT]\n{screen}\n\n"
        "Answer now: What is the user working on? What's on screen?\n"
    )


def explain(llm: LLM, snap: ContextSnapshot, max_tokens: int = 180) -> Dict:
    """Single-shot explanation."""
    return llm.generate(build_explain_prompt(snap), max_tokens)


def explain_stream(llm: LLM, snap: ContextSnapshot, on_token: Callable[[str], None],
                   max_tokens: int = 180) -> Dict:
    """Streaming explanation; returns the final result."""
    prompt = build_explain_prompt(snap)
    text = llm.stream(prompt, on_token, max_tokens=max_tokens)
    return {"text": text, "stats": {}}