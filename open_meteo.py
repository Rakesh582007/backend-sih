"""
GiriKavach — Open-Meteo Pre-Arm Layer
=======================================

WHAT THIS IS
An opportunistic forecast poller. When a storm is inbound, it raises the
system's SENSITIVITY ahead of time, so the ground sensors respond faster
when the rain actually arrives.

WHAT IT IS NOT — the critical design rule
It never raises a risk level, never fires a siren, and never blocks
anything. If Open-Meteo is unreachable (no internet — which is the
normal state in a hill village during a storm, and exactly when this
matters most), the system behaves precisely as if this module did not
exist. Everything degrades to the local sensor path, which is the path
that actually has to work.

Concretely: a pre-arm can lower `consecutive_to_escalate` from 3 to 2,
meaning the filter confirms an escalation slightly faster. It cannot
create an escalation that the sensors did not justify.

WHY OPEN-METEO
Free, no API key, no signup. Terms allow <10,000 calls/day non-commercial.
At one poll every 30 minutes that is 48 calls/day — three orders of
magnitude inside the limit.

VERIFY BEFORE THE DEMO
Variable names in the Open-Meteo API are versioned and occasionally
change. Run `python open_meteo.py` on a connected machine and confirm
the response parses before relying on it. If a variable name has moved,
the failure is loud (KeyError logged, pre-arm disabled) rather than
silent.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT_SECONDS = 10

# Poll cadence. 30 min is plenty — forecast models update hourly at best,
# so polling faster just wastes the quota and the village's bandwidth.
POLL_INTERVAL_MINUTES = 30

# Thresholds for declaring an inbound storm. These are intentionally
# conservative: pre-arming too eagerly means running in a jumpy state
# most of the monsoon, which defeats the point.
HEAVY_RAIN_MM_HR = 15.0        # IMD "heavy rainfall" is ~15mm/hr scale
LOOKAHEAD_HOURS = 6


@dataclass
class PreArmState:
    """Result of a poll. `active` False is always a safe state."""
    active: bool
    reason: str
    peak_forecast_mm_hr: float = 0.0
    hours_until_peak: int | None = None
    fetched_at: str | None = None
    # What the filter should do differently while pre-armed.
    consecutive_to_escalate: int = 3     # normal value from schema.json
    source: str = "open-meteo"


def _safe_state(reason: str) -> PreArmState:
    """Every failure path returns this. Fail closed, not open."""
    return PreArmState(active=False, reason=reason, consecutive_to_escalate=3)


def fetch_prearm(latitude: float, longitude: float) -> PreArmState:
    """Poll Open-Meteo. NEVER raises — returns a safe state on any error."""
    params = {
        "latitude": f"{latitude:.4f}",
        "longitude": f"{longitude:.4f}",
        "hourly": "precipitation",
        "forecast_days": 2,
        "timezone": "UTC",
    }
    url = f"{FORECAST_URL}?{urllib.parse.urlencode(params)}"

    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as r:
            data = json.loads(r.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as e:
        # This is the EXPECTED path during a storm/outage. Not an error
        # worth escalating — just no pre-arm this cycle.
        return _safe_state(f"forecast unavailable ({type(e).__name__}) — local sensors unaffected")

    try:
        hourly = data["hourly"]
        times = hourly["time"]
        precip = hourly["precipitation"]
    except (KeyError, TypeError):
        # Loud, not silent — if the API shape changed we want to know.
        return _safe_state("forecast response shape unexpected — variable names may have changed")

    now = datetime.now(timezone.utc)
    horizon = now + timedelta(hours=LOOKAHEAD_HOURS)

    peak = 0.0
    peak_hour_offset = None
    for t_str, mm in zip(times, precip):
        if mm is None:
            continue
        try:
            t = datetime.fromisoformat(t_str).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if now <= t <= horizon and mm > peak:
            peak = float(mm)
            peak_hour_offset = int((t - now).total_seconds() // 3600)

    fetched = now.isoformat(timespec="seconds")

    if peak >= HEAVY_RAIN_MM_HR:
        return PreArmState(
            active=True,
            reason=(
                f"heavy rain forecast: {peak:.0f}mm/hr in ~{peak_hour_offset}h — "
                f"sensitivity raised, escalation confirms in 2 readings instead of 3"
            ),
            peak_forecast_mm_hr=peak,
            hours_until_peak=peak_hour_offset,
            fetched_at=fetched,
            consecutive_to_escalate=2,
        )

    return PreArmState(
        active=False,
        reason=f"no heavy rain forecast in next {LOOKAHEAD_HOURS}h (peak {peak:.1f}mm/hr)",
        peak_forecast_mm_hr=peak,
        hours_until_peak=peak_hour_offset,
        fetched_at=fetched,
        consecutive_to_escalate=3,
    )


if __name__ == "__main__":
    # Wayanad, Kerala — swap for your deployment site.
    state = fetch_prearm(11.4654, 76.1358)
    print(f"pre-arm active : {state.active}")
    print(f"reason         : {state.reason}")
    print(f"peak forecast  : {state.peak_forecast_mm_hr} mm/hr")
    print(f"escalate after : {state.consecutive_to_escalate} readings")
    print(f"fetched at     : {state.fetched_at}")
