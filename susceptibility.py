"""
GiriKavach — Susceptibility score hook (Phase 5 PLACEHOLDER)
================================================================

STATUS: NOT BUILT. This is a wiring point, not a working feature.

backend/CLAUDE.md describes a Phase 5 susceptibility model — a static,
per-location "how dangerous is this hillside in general" score (0-1),
trained once offline on historical landslide records + terrain data,
that would multiply into live sensitivity: `final_risk = susceptibility
× live_trigger_level`. Checked before merging anything on top of it:
none of susceptibility.py, forecast.py, alerts.py, or
models/susceptibility_model.pkl actually exist yet — only the plan in
CLAUDE.md does.

WHY THIS STUB EXISTS ANYWAY
The merge plan calls for wiring susceptibility in as a FilterConfig
threshold adjustment, same pattern as open_meteo.py's pre-arm layer —
IF it turns out to be advisory (a multiplier). CLAUDE.md is explicit
that it is: a static score that adjusts sensitivity, never a second
vote on should_sound_siren. So the hook is built now, wired to a
neutral no-op, ready for a real model to drop in later without anyone
needing to touch main.py's decision path again.

THE ONE RULE THIS FILE MUST NEVER BREAK
Exactly like open_meteo.py's PreArmState: a failure, a missing model,
or NEUTRAL_SCORE must always be a SAFE, INERT default — this can only
ever adjust how many consecutive readings confirm an escalation
(FilterConfig.consecutive_to_escalate), never create an escalation the
sensors didn't justify, and never touch should_sound_siren directly.
"""

from __future__ import annotations

from dataclasses import dataclass

# 1.0 = neutral. Multiplying live sensitivity by 1.0 changes nothing —
# this is the only value this module will ever produce until a real
# trained model (models/susceptibility_model.pkl) exists to load.
NEUTRAL_SCORE = 1.0


@dataclass
class SusceptibilityState:
    score: float
    reason: str
    consecutive_to_escalate_override: int | None = None


def _safe_state(reason: str) -> SusceptibilityState:
    return SusceptibilityState(score=NEUTRAL_SCORE, reason=reason)


def load_susceptibility(node_id: str) -> SusceptibilityState:
    """Would look up a per-node susceptibility score from the trained
    Phase 5 model. Always returns the neutral/no-op state right now —
    there is no trained model to load. See module docstring.

    When Phase 5 exists: load models/susceptibility_model.pkl once at
    startup (same pattern as main.py's flood/landslide models), look up
    this node_id's precomputed score, and map a HIGH score to a LOWER
    consecutive_to_escalate (same shape as open_meteo.py's pre-arm:
    more sensitive on historically dangerous ground, never a new
    decision path).
    """
    return _safe_state("Phase 5 susceptibility model not yet built — neutral (no adjustment)")
