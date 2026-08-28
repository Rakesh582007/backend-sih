"""GiriKavach dashboard: a pure view over the teammate's FastAPI decision
backend. Polls GET /status (latest decision), GET /history?limit=N (chart
data), and GET /health (service/model load state), and renders the full
decision object exactly as schema.json's `decision_object` defines it.

This dashboard computes NO risk itself — no thresholds, no siren logic, no
false-positive filtering. If the backend is unreachable, or a response is
missing fields this dashboard expects, that's shown as a clear warning, never
papered over or guessed at.

NOTE on contract.py: backend/ml/contract.py (the shared schema loader per
CLAUDE.md §6) lives on the `feat/ml-tuning` branch, not this one, so it isn't
importable here yet. Until the branches merge, this file loads schema.json
directly using the same read-don't-hardcode approach contract.py uses — same
single source of truth (the live schema file), just two call sites for now.
Once merged, switch the loader calls below to `from ml.contract import ...`
and delete the local `_load_schema`/`_risk_tiers`/etc. helpers.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")
POLL_SECONDS = 5
HISTORY_LIMIT = 200

# backend/dashboard/streamlit_app.py -> repo root is two directories up.
SCHEMA_PATH = Path(__file__).resolve().parent.parent.parent / "schema.json"

DEPTH_FIELDS = [
    ("soil_moisture_10cm_pct", 10),
    ("soil_moisture_30cm_pct", 30),
    ("soil_moisture_60cm_pct", 60),
    ("soil_moisture_100cm_pct", 100),
    ("soil_moisture_150cm_pct", 150),
]

RISK_STYLE = {
    "NORMAL": {"color": "#2e7d32", "bg": "#e8f5e9"},  # green
    "WATCH": {"color": "#8d6d00", "bg": "#fff8e1"},  # amber
    "ALERT": {"color": "#e65100", "bg": "#fff3e0"},  # orange
    "CRITICAL": {"color": "#c62828", "bg": "#ffebee"},  # red
}

st.set_page_config(page_title="GiriKavach Monitor", page_icon=":material/landslide:", layout="wide")


# --------------------------------------------------------------------------
# schema access (see NOTE at top of file re: contract.py)
# --------------------------------------------------------------------------

@st.cache_data(ttl=None)
def _load_schema() -> dict:
    with SCHEMA_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def _risk_tiers() -> dict:
    return {k: v for k, v in _load_schema()["risk_tiers"].items() if not k.startswith("_")}


def _stale_after_seconds() -> int:
    return _load_schema()["heartbeat"]["stale_after_seconds"]


def _decision_object_keys() -> list[str]:
    return [k for k in _load_schema()["decision_object"].keys() if not k.startswith("_")]


# Fail loudly at import if the tier strings this dashboard styles don't match
# the schema's tier strings — a renamed/added tier would otherwise silently
# render with the NORMAL fallback color instead of raising anything.
_schema_tiers = set(_risk_tiers().keys())
if _schema_tiers != set(RISK_STYLE.keys()):
    st.error(
        f"risk_tiers in schema.json {sorted(_schema_tiers)} no longer match the "
        f"tiers this dashboard has colors for {sorted(RISK_STYLE.keys())}. "
        f"Fix RISK_STYLE before trusting the banner below.",
        icon=":material/error:",
    )


# --------------------------------------------------------------------------
# backend fetch — each call handles its own failure so one endpoint being
# down doesn't take the others with it
# --------------------------------------------------------------------------

def _get(path: str, **params):
    """GET {BACKEND_URL}{path}. Returns (data, error) — exactly one is None."""
    try:
        resp = requests.get(f"{BACKEND_URL}{path}", params=params or None, timeout=3)
    except requests.exceptions.RequestException:
        return None, f"Can't reach the backend at {BACKEND_URL} — is it running?"
    if not resp.ok:
        return None, f"Backend returned {resp.status_code} for {path}: {resp.text[:200]}"
    try:
        return resp.json(), None
    except ValueError:
        return None, f"Backend response for {path} wasn't valid JSON."


def fetch_status():
    return _get("/status")


def fetch_history(limit: int = HISTORY_LIMIT):
    return _get("/history", limit=limit)


def fetch_health():
    return _get("/health")


def parse_ts(ts: str) -> datetime | None:
    """Parses the decision object's ISO 8601 timestamp (schema says +05:30 on
    the object the backend returns). Returns None rather than raising on
    anything unparseable."""
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


# --------------------------------------------------------------------------
# render helpers
# --------------------------------------------------------------------------

def moisture_color(pct: float) -> str:
    """Light blue (dry) -> dark blue (saturated), for the depth-profile strip."""
    frac = max(0.0, min(100.0, pct)) / 100.0
    start, end = (227, 242, 253), (13, 71, 161)  # #e3f2fd -> #0d47a1
    r, g, b = (round(start[i] + (end[i] - start[i]) * frac) for i in range(3))
    return f"rgb({r},{g},{b})"


def readable_text_color(pct: float) -> str:
    return "#ffffff" if pct >= 55 else "#0d1b2a"


def render_risk_banner(decision: dict):
    level = decision.get("risk_level")
    style = RISK_STYLE.get(level)
    if style is None:
        st.error(f"Unknown risk_level {level!r} — not one of {sorted(RISK_STYLE)}.", icon=":material/error:")
        style = RISK_STYLE["NORMAL"]
    st.markdown(
        f'<div style="background:{style["bg"]};border:1px solid {style["color"]}33;'
        f'border-radius:10px;padding:20px 24px;">'
        f'<div style="font-size:1.8rem;font-weight:700;color:{style["color"]};line-height:1.1;">'
        f'{level or "UNKNOWN"}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_probability_metrics(decision: dict):
    col1, col2 = st.columns(2)
    flood = decision.get("flood_probability")
    landslide = decision.get("landslide_probability")
    with col1:
        st.metric("Flood probability", f"{flood:.1f}%" if isinstance(flood, (int, float)) else "n/a")
    with col2:
        st.metric("Landslide probability", f"{landslide:.1f}%" if isinstance(landslide, (int, float)) else "n/a")


def render_node_chip(decision: dict):
    with st.container(border=True):
        st.markdown("**Node**")
        st.caption(f"ID: {decision.get('node_id', 'unknown')}")
        ts_raw = decision.get("timestamp")
        dt = parse_ts(ts_raw) if ts_raw else None
        stale_after = _stale_after_seconds()
        if dt is None:
            st.badge("UNKNOWN", icon=":material/help:", color="gray")
            st.caption("No parseable timestamp in the decision object.")
            return
        age_s = (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds()
        if age_s > stale_after:
            st.badge("STALE", icon=":material/error:", color="red")
        else:
            st.badge("ONLINE", icon=":material/check_circle:", color="green")
        st.caption(f"Last decision: {ts_raw} ({age_s:.0f}s ago · stale after {stale_after}s)")


def render_badges(decision: dict):
    with st.container(border=True):
        st.markdown("**Flags**")
        if decision.get("should_sound_siren"):
            st.badge("SIREN", icon=":material/campaign:", color="red")
        else:
            st.badge("siren off", icon=":material/campaign:", color="green")
        if decision.get("safety_net_triggered"):
            st.badge("SAFETY NET TRIGGERED", icon=":material/warning:", color="orange")
        else:
            st.badge("safety net idle", icon=":material/shield:", color="green")


def render_reasons_and_suppressed(decision: dict):
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("**Reasons — why it escalated**")
            reasons = decision.get("reasons") or []
            if reasons:
                for r in reasons:
                    st.markdown(f"- {r}")
            else:
                st.caption("None reported.")
    with col2:
        with st.container(border=True):
            st.markdown("**Suppressed — what was rejected, and why**")
            suppressed = decision.get("suppressed") or []
            if suppressed:
                for s in suppressed:
                    st.markdown(f"- {s}")
            else:
                st.caption("None reported.")


def render_sensor_health_warnings(decision: dict):
    with st.container(border=True):
        st.markdown("**Sensor health warnings**")
        warnings_list = decision.get("sensor_health_warnings") or []
        if warnings_list:
            for w in warnings_list:
                st.warning(w, icon=":material/sensors_off:")
        else:
            st.caption("No warnings.")


def extract_depth_profile(decision: dict) -> dict[str, float] | None:
    """The frozen decision_object does NOT include the 5-depth soil profile —
    only the derived deep_soil_moisture_pct feeds the ML models, per
    schema.json. If the running backend includes the raw per-depth fields
    anyway (flat on the response, or nested under 'soil_moisture_profile'),
    show them; otherwise say so plainly instead of fabricating a profile.
    This is a known open gap between CLAUDE.md's dashboard spec and the
    frozen decision_object — flagged, not guessed around."""
    nested = decision.get("soil_moisture_profile")
    source = nested if isinstance(nested, dict) else decision
    profile = {field: source[field] for field, _ in DEPTH_FIELDS
               if isinstance(source.get(field), (int, float))}
    return profile or None


def render_depth_profile(decision: dict):
    with st.container(border=True):
        st.markdown("**Soil moisture depth profile**")
        profile = extract_depth_profile(decision)
        if not profile:
            st.caption(
                "Not available from `/status` right now — the frozen decision_object "
                "only carries the derived `deep_soil_moisture_pct`, not the raw "
                "5-depth profile. Showing this strip needs the backend to also expose "
                "the per-depth fields on the response; confirm with the team whether "
                "it does."
            )
            return
        rows = []
        for field, depth in DEPTH_FIELDS:
            if field not in profile:
                continue
            pct = profile[field]
            color = moisture_color(pct)
            text_color = readable_text_color(pct)
            rows.append(
                f'<div style="background:{color};color:{text_color};padding:12px 16px;'
                f'border-radius:6px;margin-bottom:5px;display:flex;justify-content:space-between;'
                f'font-size:0.9rem;">'
                f'<span>{depth} cm</span><span><b>{pct:.1f}%</b></span></div>'
            )
        st.markdown("".join(rows), unsafe_allow_html=True)
        st.caption("Dark = saturated, light = dry — the wetting front as it moves down the profile")


def render_service_health(health: dict | None, health_err: str | None):
    with st.container(border=True):
        st.markdown("**Service health** (`/health`)")
        if health_err:
            st.warning(health_err, icon=":material/cloud_off:")
            return
        if not isinstance(health, dict) or not health:
            st.warning("`/health` returned an empty or unexpected response.", icon=":material/help:")
            return
        for k, v in health.items():
            st.caption(f"{k}: {v}")


def build_history_df(history: list) -> pd.DataFrame:
    rows = []
    for d in history:
        if not isinstance(d, dict):
            continue
        ts = parse_ts(d.get("timestamp")) if d.get("timestamp") else None
        if ts is None:
            continue
        row = {"timestamp": ts}
        for key in (
            "flood_probability", "landslide_probability",
            "stream_depth_cm", "slope_pitch_deg", "rate_of_rise_cm_min",
        ):
            val = d.get(key)
            if isinstance(val, (int, float)):
                row[key] = val
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("timestamp").set_index("timestamp")


def render_probability_trend(df: pd.DataFrame):
    with st.container(border=True):
        st.markdown("**Flood / landslide probability over time**")
        cols = [c for c in ("flood_probability", "landslide_probability") if c in df.columns]
        if df.empty or not cols:
            st.caption("Not enough history yet.")
            return
        st.line_chart(df[cols], y_label="Probability (%)")


def render_raw_trend(df: pd.DataFrame):
    with st.container(border=True):
        st.markdown("**Water level / tilt — raw sensor trend**")
        cols = [c for c in ("stream_depth_cm", "slope_pitch_deg", "rate_of_rise_cm_min") if c in df.columns]
        if df.empty or not cols:
            st.caption(
                "Not available from `/history` right now — the frozen decision_object "
                "carries flood/landslide probabilities, not the raw stream_depth_cm / "
                "slope_pitch_deg / rate_of_rise_cm_min readings. Confirm with the team "
                "whether `/history` records also carry the raw readings alongside each "
                "decision."
            )
            return
        st.line_chart(df[cols])


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

st.title("GiriKavach — flood / landslide monitor")
st.caption(f"Backend: {BACKEND_URL}")


@st.fragment(run_every=f"{POLL_SECONDS}s")
def render_dashboard():
    status, status_err = fetch_status()
    st.caption(
        f"Polling every {POLL_SECONDS}s · last refreshed "
        f"{datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC"
    )

    if status_err:
        st.error(status_err, icon=":material/cloud_off:")
        st.info(
            "No decision to show — this dashboard doesn't compute risk itself, "
            "it only renders what the backend returns."
        )
        return

    if not isinstance(status, dict) or not status:
        st.warning("`/status` returned an empty or unexpected response.", icon=":material/help:")
        return

    missing = [k for k in _decision_object_keys() if k not in status]
    if missing:
        st.warning(
            f"`/status` response is missing expected field(s): {', '.join(missing)}. "
            f"Rendering what's present.",
            icon=":material/warning:",
        )

    render_risk_banner(status)
    render_probability_metrics(status)

    with st.container(horizontal=True):
        render_node_chip(status)
        render_badges(status)

    if status.get("node1_link_fresh") is False:
        st.error(
            "node1_link_fresh is False — landslide inference was SKIPPED upstream "
            "(all its features originate at node1). This is NOT a low-risk finding.",
            icon=":material/link_off:",
        )

    render_reasons_and_suppressed(status)
    render_sensor_health_warnings(status)

    col_left, col_right = st.columns(2)
    with col_left:
        render_depth_profile(status)
    with col_right:
        health, health_err = fetch_health()
        render_service_health(health, health_err)

    st.divider()
    history, history_err = fetch_history(HISTORY_LIMIT)
    if history_err:
        st.warning(history_err, icon=":material/cloud_off:")
    elif not isinstance(history, list):
        st.warning("`/history` returned an unexpected shape (expected a list).", icon=":material/help:")
    else:
        df = build_history_df(history)
        col_a, col_b = st.columns(2)
        with col_a:
            render_probability_trend(df)
        with col_b:
            render_raw_trend(df)


render_dashboard()
