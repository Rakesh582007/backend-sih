"""
GiriKavach — Outgoing alert validator
========================================

The last checkpoint before ANY citizen-facing message is allowed to go
out. Catches exactly the failure modes real LLM testing produced this
session: a hallucinated place name, a silently-dropped drill prefix,
and invented numbers not present in the actual input.

Deliberately simple, deterministic string/regex checks — no LLM, no
fuzzy matching, no judgement calls. A validator that itself needs
interpretation is not a validator; it's another thing that can fail
silently.
"""

from __future__ import annotations

import re

from alert_templates import DRILL_PREFIX

MIN_LENGTH = 20
MAX_LENGTH = 400


def check_alert_text(
    text: str,
    expected_place: str,
    is_exercise: bool,
    allowed_numbers: set[str] | None = None,
) -> tuple[bool, list[str]]:
    """Validate outgoing alert text against the four rules every
    citizen-facing message must satisfy.

    Returns (passed, failures). `failures` is a list of human-readable
    reasons and is empty if and only if `passed` is True — `passed` is
    the authoritative result, `failures` is for logging/debugging why.
    """
    failures: list[str] = []
    allowed_numbers = allowed_numbers or set()

    if expected_place not in text:
        failures.append(
            f"expected place {expected_place!r} not found verbatim in text"
        )

    if is_exercise and DRILL_PREFIX.strip() not in text:
        failures.append(
            f"is_exercise=True but drill prefix {DRILL_PREFIX.strip()!r} is missing"
        )

    found_numbers = set(re.findall(r"\d+", text))
    unexpected = found_numbers - allowed_numbers
    if unexpected:
        failures.append(
            f"unexpected number(s) not present in input slots: {sorted(unexpected)}"
        )

    if not (MIN_LENGTH <= len(text) <= MAX_LENGTH):
        failures.append(
            f"length {len(text)} outside allowed range [{MIN_LENGTH}, {MAX_LENGTH}]"
        )

    return (len(failures) == 0, failures)
