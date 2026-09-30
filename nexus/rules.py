"""Deterministic alert engine - no LLM required.

Detects risks directly from the text visible on screen:
  * API keys & secrets
  * Dangerous commands
  * Personal data (Aadhaar / PAN / credit cards via the Luhn check)

All detection predicates are pure functions so they're trivial to unit test.
The ``RuleEngine`` layer adds deduplication (cooldown) so the same alert is not
shown repeatedly.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .utils import looks_high_entropy, mask_secret, setup_logging

log = setup_logging(__name__)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
SEVERITIES = ("critical", "high", "medium", "low")


@dataclass
class Alert:
    """A single, deduplicated alert about something on screen."""

    category: str  # secret | dangerous_command | personal_data
    severity: str  # critical | high | medium | low
    matched: str  # already-masked snippet
    suggestion: str
    raw: str = ""  # never surfaced to the UI; used for dedup
    created_at: float = field(default_factory=time.time)

    @property
    def signature(self) -> str:
        return f"{self.category}:{self.matched[:40]}"


# ---------------------------------------------------------------------------
# Registers of patterns
# ---------------------------------------------------------------------------
_SECRET_PATTERNS: List[Tuple[str, str, str, str]] = [
    ("aws", r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b", "high",
     "AWS credential detected. Rotate the key and remove it from your source."),
    ("google", r"AIza[0-9A-Za-z_-]{35}\b", "high",
     "Google API key detected. Revoke and replace it, store it in env/secrets."),
    ("openai", r"sk-(?:proj-)?[A-Za-z0-9]{16,}\b", "critical",
     "OpenAI API key visible. Revoke immediately and switch to an env var."),
    ("github", r"\b(?:ghp_|gho_|ghu_|github_pat_)[A-Za-z0-9_]{20,}\b", "high",
     "GitHub token on screen. Rotate it and avoid pasting secrets into files."),
    ("slack", r"xox[baprs]-[A-Za-z0-9-]{10,}\b", "high",
     "Slack token detected. Revoke and use a vault / env var."),
    ("stripe", r"\b(?:sk|rk)_live_[0-9a-zA-Z]{16,}\b", "high",
     "Stripe live key exposed. Rotate immediately (this is a live, not test, key)."),
    ("jwt", r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b",
     "high", "A JWT / signed token is visible. Be careful sharing or committing it."),
    ("private_key", r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----",
     "critical", "A private key header is on screen. Never share or commit it."),
]

_ASSIGN_RE = re.compile(
    r"\b(?:password|passwd|pwd|secret|token|api[_-]?key|apikey|client[_-]?secret)"
    r"\s*[=:]\s*[\"']?([^\s\"',;]+)",
    re.IGNORECASE,
)

_DANGER_PATTERNS: List[Tuple[str, str, str, str]] = [
    ("rm_force", r"\brm\s+(-[a-z]*r[a-z]*f|-[a-z]*f[a-z]*r)\b", "high",
     "Force-recursive delete detected. Double-check the target path before running."),
    ("rm_root", r"\brm\s+(?:-rf\s+)?/\s*", "critical",
     "Deleting the filesystem root - this is almost certainly a mistake or a prank."),
    ("del_force", r"\b(?:del|rd|rmdir)\s+/[a-z]*[sq][a-z]*\b", "high",
     "Recursive force delete on a whole folder. Verify before continuing."),
    ("format", r"\bformat\s+[a-z]:", "high",
     "Formatting a drive partition detected. This wipes data irreversibly."),
    ("drop_table", r"\bDROP\s+(?:TABLE|DATABASE)\b", "high",
     "Dropping a table/database destroys data. Confirm you are on the right instance."),
    ("delete_no_where", r"\bDELETE\s+FROM\s+\w+(?![\s\S]*\bWHERE\b)", "critical",
     "DELETE without a WHERE clause - this affects every row. Add a WHERE."),
    ("force_push", r"\bgit\s+push\s+(?:--force|-f)\b", "high",
     "Force push rewrites history and can destroy work on the remote. Be certain."),
    ("chmod_777", r"\bchmod\s+(?:-R\s+|--recursive\s+)?777\b", "medium",
     "chmod 777 grants permissions to everyone. Prefer narrower permissions."),
    ("pipe_sh", r"\b(?:curl|wget)\b[^\n|]*\|\s*(?:sudo\s+)?(?:bash|sh)\b", "high",
     "Piping remote code straight into a shell executes it without review. Avoid it."),
    ("remove_item", r"\bRemove-Item\s+(?:-Recurse\s+)?(?:-Force\s*)?\b", "high",
     "PowerShell force-recursive remove detected. Confirm the target path."),
    ("mkfs", r"\bmkfs\.\w+", "high", "Filesystem creation wipes the target device."),
]

# Personal data
_PAN_RE = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
_AADHAAR_RE = re.compile(r"(?<!\d)([2-9]\d{11})(?!\d)")
_CARD_CANDIDATE_RE = re.compile(r"(?<!\d)(\d{13,19})(?!\d)")


def luhn_ok(digits: str) -> bool:
    """Luhn checksum check - the standard card-number validation."""

    def _checksum(num: str) -> bool:
        total = 0
        reverse = num[::-1]
        for i, ch in enumerate(reverse):
            d = ord(ch) - 48
            if i % 2 == 1:
                d *= 2
                if d > 9:
                    d -= 9
            total += d
        return total % 10 == 0

    if not digits.isdigit() or len(digits) < 13:
        return False
    return _checksum(digits)


# ---------------------------------------------------------------------------
# Pure detection predicates (unit-testable, no state)
# ---------------------------------------------------------------------------
def detect_secrets(text: str) -> List[Alert]:
    """Find API keys, tokens and private key material in ``text``."""
    alerts: List[Alert] = []
    if not text:
        return alerts
    for name, pattern, sev, suggestion in _SECRET_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            raw = m.group(0)
            alerts.append(Alert(
                category="secret", severity=sev,
                matched=mask_secret(raw), suggestion=suggestion, raw=raw,
            ))
    # assignment style: password=..., token=...
    for m in _ASSIGN_RE.finditer(text):
        value = m.group(1)
        if looks_high_entropy(value) and len(value) >= 8:
            alerts.append(Alert(
                category="secret", severity="medium",
                matched=f"{m.group(0)[:20]}=",  # never echo the value
                suggestion="Secrets assigned inline are at risk. Move to env/secrets.",
                raw=value,
            ))
    return alerts


def detect_dangerous(text: str) -> List[Alert]:
    """Find destructive or risky shell / SQL commands."""
    alerts: List[Alert] = []
    if not text:
        return alerts
    for name, pattern, sev, suggestion in _DANGER_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            alerts.append(Alert(
                category="dangerous_command", severity=sev,
                matched=m.group(0)[:48], suggestion=suggestion, raw=m.group(0),
            ))
    return alerts


def detect_personal(text: str) -> List[Alert]:
    """Find Aadhaar-like IDs, PAN numbers and card numbers (Luhn-valid)."""
    alerts: List[Alert] = []
    if not text:
        return alerts
    for m in _PAN_RE.finditer(text):
        raw = m.group(0)
        alerts.append(Alert(
            category="personal_data", severity="medium",
            matched="PAN ****" + raw[-1],
            suggestion="An income-tax PAN is sensitive personal data. Redact it.",
            raw=raw,
        ))
    for m in _AADHAAR_RE.finditer(text):
        raw = m.group(1)
        # Avoid false positives on arbitrary 12-digit numbers (e.g. phone chunks).
        if luhn_ok(raw) or alive_verhoeff(raw):
            alerts.append(Alert(
                category="personal_data", severity="high",
                matched="Aadhaar ID ********" + raw[-4:],
                suggestion="Aadhaar is PII under the Aadhaar Act. Never commit/store it.",
                raw=raw,
            ))
    for m in _CARD_CANDIDATE_RE.finditer(text):
        raw = m.group(1)
        if luhn_ok(raw):
            alerts.append(Alert(
                category="personal_data", severity="high",
                matched="Card **** **** **** " + raw[-4:],
                suggestion="A valid card number is on screen. Securely wipe it.",
                raw=raw,
            ))
    return alerts
# Verhoeff checksum used by the real Aadhaar number (optional, improves accuracy).
_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6], [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2], [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 7, 0, 8], [4, 2, 8, 6, 5, 7, 3, 0, 1, 9],
    [2, 6, 9, 4, 7, 0, 5, 1, 3, 8], [6, 3, 4, 5, 9, 8, 7, 2, 0, 1],
    [3, 7, 4, 1, 8, 5, 9, 6, 2, 0], [7, 3, 2, 8, 6, 4, 1, 9, 5, 0],
]
_VERHOEFF_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def alive_verhoeff(num: str) -> bool:
    """Verhoeff checksum check (used by real Aadhaar numbers)."""
    try:
        c = 0
        for i, ch in enumerate(reversed(num)):
            c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(ch)]]
        return _VERHOEFF_INV[c] == 0
    except (ValueError, IndexError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Stateful engine with deduplication / cooldown
# ---------------------------------------------------------------------------
class RuleEngine:
    """Evaluates screen text and returns NEW alerts respecting a cooldown."""

    def __init__(self, cooldown: float = 30.0, suppress: Optional[List[str]] = None):
        self.cooldown = cooldown
        self.suppress = set(suppress or [])
        self._last_seen: Dict[str, float] = {}

    def evaluate(self, text: str) -> List[Alert]:
        """Return deduplicated alerts for ``text`` (recurring ones suppressed)."""
        new_alerts: List[Alert] = []
        for alert in (
            detect_secrets(text) + detect_dangerous(text) + detect_personal(text)
        ):
            if alert.category in self.suppress:
                continue
            now = alert.created_at
            last = self._last_seen.get(alert.signature)
            if last is not None and (now - last) < self.cooldown:
                continue
            self._last_seen[alert.signature] = now
            new_alerts.append(alert)
        return new_alerts