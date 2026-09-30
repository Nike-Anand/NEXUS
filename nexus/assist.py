"""Assist: suggest next steps grounded in the current screen context.

Useful as a hotkey-triggered "what should I do next?" or a typed request. It is
text-only by design and never performs actions - the (optional, later) Act
module is where user-approved actions live.
"""
from __future__ import annotations

from typing import Dict

from .context import ContextSnapshot
from .llm import LLM

SYSTEM = (
    "You are NEXUS, a private on-device assistant. Suggest 2-4 concrete next "
    "steps for the user's current task, based only on the screen context. "
    "Be specific and safe. Do not run anything yourself."
)


def build_assist_prompt(snap: ContextSnapshot, request: str = "") -> str:
    screen = (snap.text or "")[:1500]
    base = (
        f"{SYSTEM}\n\nApp type: {snap.app_type}\nActive window: "
        f"{snap.active_window}\n\n[SCREEN CONTEXT]\n{screen}\n\n"
    )
    if request:
        return base + f"User request: {request}\nSuggest next steps.\n"
    return base + "Suggest next steps for what the user is doing.\n"


def assist(llm: LLM, snap: ContextSnapshot, request: str = "",
           max_tokens: int = 200) -> Dict:
    """Suggest next steps grounded in the current screen context."""
    return llm.generate(build_assist_prompt(snap, request), max_tokens)