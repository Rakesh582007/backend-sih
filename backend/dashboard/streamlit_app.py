"""Phase 6 dashboard: polls GET /status/{village_id} on the FastAPI backend and renders
the live picture for one node — risk verdict, node health, water level/rate, tilt trace,
and the soil depth profile (the "wetting front" money-shot the manual calls out).

This dashboard has no database of its own: /status only ever returns the latest reading,
so the tilt/water-level traces are built client-side by accumulating each poll into
st.session_state, capped to a rolling window.
"""

from datetime import datetime, timezone

import pandas as pd
import requests
import streamlit as st

BACKEND_URL = "http://localhost:8000"
POLL_SECONDS = 5
HISTORY_LIMIT = 200  # rolling cap per node so session memory can't grow unbounded

DEPTH_CHANNELS = [("soil_1", 10), ("soil_2", 30), ("soil_3", 60), ("soil_4", 100), ("soil_5", 150)]

RISK_STYLE = {
    "NORMAL": {"color": "#2e7d32", "bg": "#e8f5e9"},
    "WATCH": {"color": "#8d6d00", "bg": "#fff8e1"},
    "ALERT": {"color": "#e65100", "bg": "#fff3e0"},
    "CRITICAL": {"color": "#c62828", "bg": "#ffebee"},
}

st.set_page_config(page_title="Ridge & Valley Monitor", page_icon=":material/landslide:", layout="wide")


# --------------------------------------------------------------------------
# data fetch
# --------------------------------------------------------------------------

def fetch_status(village_id: str):
    """Returns (data, error_message) — exactly one of them is not None."""
    try:
        resp = requests.get(f"{BACKEND_URL}/status/{village_id}", timeout=3)
    except requests.exceptions.RequestException:
        return None, f"Can't reach the backend at {BACKEND_URL} — is it running?"
    if resp.status_code == 404:
        return None, f"No readings yet for node '{village_id}'."
    if not resp.ok:
        return None, f"Backend returned {resp.status_code}: {resp.text}"
    return resp.json(), None


def record_history(village_id: str, data: dict):
    """Append this poll's reading to the node's rolling history if it's new data —
    /status can return the same reading across several polls between real sensor pings."""
    history = st.session_state.setdefault("history", {}).setdefault(village_id, [])
    if not history or history[-1]["timestamp"] != data["timestamp"]:
        history.append({
            "timestamp": data["timestamp"],
            "tilt": data["tilt"],
            "water_level": data["water_level"],
        })
        del history[: len(history) - HISTORY_LIMIT]
    return history


# --------------------------------------------------------------------------
# small render helpers
# --------------------------------------------------------------------------

def moisture_color(pct: float) -> str:
    """Light blue (dry) -> dark blue (saturated), for the depth-profile strip."""
    frac = max(0.0, min(100.0, pct)) / 100.0
    start, end = (227, 242, 253), (13, 71, 161)  # #e3f2fd -> #0d47a1
    r, g, b = (round(start[i] + (end[i] - start[i]) * frac) for i in range(3))
    return f"rgb({r},{g},{b})"


def readable_text_color(pct: float) -> str:
    return "#ffffff" if pct >= 55 else "#0d1b2a"


def render_risk_banner(data: dict):
    level = data["level"]
    style = RISK_STYLE.get(level, RISK_STYLE["NORMAL"])
    st.markdown(
        f'<div style="background:{style["bg"]};border:1px solid {style["color"]}33;'
        f'border-radius:10px;padding:20px 24px;">'
        f'<div style="font-size:1.8rem;font-weight:700;color:{style["color"]};line-height:1.1;">'
        f'{level}</div>'
        f'<div style="color:{style["color"]};opacity:0.85;font-size:0.9rem;">'
        f'flood: {data["flood_level"]} &nbsp;·&nbsp; landslide: {data["landslide_level"]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_node_health(data: dict):
    with st.container(border=True):
        st.markdown("**Node health**")
        if data["node_dead"]:
            st.badge("DEAD", icon=":material/error:", color="red")
            st.caption("No ingest in over 15s")
        else:
            st.badge("ONLINE", icon=":material/check_circle:", color="green")
        st.caption(f"Last reading: {data['timestamp']}")
        if data["flags"]:
            st.caption(f":material/health_and_safety: self-healing: {data['flags']}")
        st.caption(f":material/battery_full: battery {data['battery']:.2f} V")


def render_water_metrics(data: dict):
    with st.container(border=True):
        st.markdown("**Water level**")
        dh_dt = data["dh_dt_cm_per_s"]
        delta = f"{dh_dt:+.2f} cm/s" if dh_dt is not None else "not enough data yet"
        st.metric("Ultrasonic reading", f"{data['water_level']:.1f} cm", delta=delta, border=False)


def render_landslide_metrics(data: dict):
    with st.container(border=True):
        st.markdown("**Landslide indicators**")
        tilt_rate = data["tilt_rate_deg_per_hr"]
        delta = f"{tilt_rate:+.2f} °/hr" if tilt_rate is not None else None
        st.metric("Tilt", f"{data['tilt']:.2f}°", delta=delta, border=False)
        tilt_delta = data["tilt_delta_deg"]
        theta = data["theta_deep_pct"]
        st.caption(
            f"Δ from baseline: {tilt_delta:.2f}°" if tilt_delta is not None else "Δ from baseline: n/a (baseline not established)"
        )
        st.caption(f"Deep soil θ (S4×0.4 + S5×0.6): {theta:.1f}%" if theta is not None else "Deep soil θ: n/a")
        st.caption(f"Rain duration: {data['rain_duration_min']:.0f} min")


def render_depth_profile(data: dict):
    with st.container(border=True):
        st.markdown("**Soil moisture depth profile**")
        rows = []
        for channel, depth in DEPTH_CHANNELS:
            pct = data[channel]
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


def render_tilt_trace(history: list):
    with st.container(border=True):
        st.markdown("**Tilt trace**")
        if len(history) < 2:
            st.caption("Collecting readings — the trace fills in as the dashboard keeps polling.")
            return
        df = pd.DataFrame(history)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
        st.line_chart(df[["tilt"]], y_label="Tilt (deg)")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

st.title("Ridge & Valley flood/landslide monitor")
village_id = st.text_input("Village / node ID", value="village_1")


@st.fragment(run_every=f"{POLL_SECONDS}s")
def render_dashboard(village_id: str):
    data, error = fetch_status(village_id)

    st.caption(f"Polling every {POLL_SECONDS}s · last refreshed {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC")

    if error:
        st.warning(error, icon=":material/error:")
        return

    history = record_history(village_id, data)

    render_risk_banner(data)
    st.space("small")

    with st.container(horizontal=True):
        render_node_health(data)
        render_water_metrics(data)
        render_landslide_metrics(data)

    col_left, col_right = st.columns(2)
    with col_left:
        render_depth_profile(data)
    with col_right:
        render_tilt_trace(history)


render_dashboard(village_id)
