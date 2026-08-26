"""Phase 4 risk engine: deterministic trigger matrices for flood and landslide.

No ML here on purpose (see CLAUDE.md) — every number this module returns is something
a person can recompute by hand from the trigger matrix and argue about, which is the
point for a safety system. The susceptibility model (Phase 5) is a separate multiplier
applied on top of whatever `evaluate_risk()` returns; it does not belong in this file.
"""

from datetime import datetime, timedelta
from statistics import mean, median

from sqlalchemy.orm import Session

from db import SensorReading

LEVELS = ["NORMAL", "WATCH", "ALERT", "CRITICAL"]

# --- shared regression window ---
REGRESSION_WINDOW_SECONDS = 60   # per manual: fit the slope over the trailing 60s, not 2 points
MEDIAN_FILTER_WINDOW = 5         # per manual: denoise with a rolling median before fitting

# --- flood thresholds ---
H_BANKFULL_CM = 100.0            # placeholder — replace with the real per-site bankfull height
FLOOD_WATCH_DHDT_CM_S = 0.5
FLOOD_ALERT_DHDT_CM_S = 1.5
FLOOD_CRITICAL_DHDT_CM_S = 3.0
FLOOD_CRITICAL_LEVEL_FRACTION = 0.8

# --- landslide thresholds ---
TILT_BASELINE_SAMPLE_SIZE = 200  # install-baseline = mean tilt of the first N readings ever seen
DEEP_SOIL_WEIGHTS = {"soil_4": 0.4, "soil_5": 0.6}

LANDSLIDE_WATCH_RAIN_MIN = 30
LANDSLIDE_WATCH_THETA_PCT = 70.0

LANDSLIDE_ALERT_RAIN_MIN = 60
LANDSLIDE_ALERT_THETA_PCT = 85.0
LANDSLIDE_ALERT_TILT_DELTA_DEG = 0.5

LANDSLIDE_CRITICAL_THETA_PCT = 85.0
LANDSLIDE_CRITICAL_TILT_DELTA_DEG = 2.0
LANDSLIDE_CRITICAL_TILT_RATE_DEG_HR = 1.0


# --------------------------------------------------------------------------
# shared math helpers
# --------------------------------------------------------------------------

def _rolling_median(values: list, window: int = MEDIAN_FILTER_WINDOW) -> list:
    """Trailing (causal) rolling median: each point is smoothed using only itself and
    up to `window`-1 points *before* it. A centered filter would need readings from
    after 'now' to smooth the most recent (i.e. most urgent) point — which don't
    exist yet in a live system — and would silently lag the alert behind reality."""
    return [median(values[max(0, i - window + 1):i + 1]) for i in range(len(values))]


def _regression_slope_per_second(points: list) -> float | None:
    """Least-squares slope of y over x (x already in seconds, ascending). None if there
    aren't at least 2 distinct-in-time points to fit a line through."""
    if len(points) < 2:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_bar = mean(xs)
    y_bar = mean(ys)
    denom = sum((x - x_bar) ** 2 for x in xs)
    if denom == 0:
        return None
    numer = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys))
    return numer / denom


CONTEXT_LIMIT = 20  # extra history fetched so the causal median filter is warmed up
                     # by the time it reaches the trailing 60s window, instead of being
                     # distorted by its own bootstrap at the window's left edge


def _slope_over_trailing_window(
    db: Session, node_id: str, now: datetime, value_of, smooth: bool,
    window_seconds: int = REGRESSION_WINDOW_SECONDS,
) -> float | None:
    """
    Least-squares slope over the trailing `window_seconds`. Fetches extra history
    *before* the window too (see CONTEXT_LIMIT) so that when `smooth` is True, the
    causal median filter has real prior readings to warm up on rather than only the
    handful of points that happen to fall inside the window itself.
    """
    context_rows = (
        db.query(SensorReading)
        .filter(SensorReading.node_id == node_id, SensorReading.timestamp <= now)
        .order_by(SensorReading.timestamp.desc())
        .limit(CONTEXT_LIMIT)
        .all()
    )
    context_rows.reverse()  # ascending
    if len(context_rows) < 2:
        return None

    raw_values = [value_of(r) for r in context_rows]
    values = _rolling_median(raw_values) if smooth else raw_values

    window_start = now - timedelta(seconds=window_seconds)
    t0 = context_rows[0].timestamp
    points = [
        ((r.timestamp - t0).total_seconds(), v)
        for r, v in zip(context_rows, values)
        if r.timestamp >= window_start
    ]
    return _regression_slope_per_second(points)


# --------------------------------------------------------------------------
# flood
# --------------------------------------------------------------------------

def _flood_dh_dt_cm_per_s(db: Session, node_id: str, now: datetime) -> float | None:
    """Median-filter the last pings, then fit a least-squares slope over the trailing
    60s — a single noisy ultrasonic ping can't swing this on its own."""
    return _slope_over_trailing_window(db, node_id, now, lambda r: r.water_level, smooth=True)


def _flood_level(dh_dt_cm_per_s: float | None, water_level_cm: float) -> str:
    if water_level_cm > FLOOD_CRITICAL_LEVEL_FRACTION * H_BANKFULL_CM:
        return "CRITICAL"
    if dh_dt_cm_per_s is None:
        return "NORMAL"
    if dh_dt_cm_per_s > FLOOD_CRITICAL_DHDT_CM_S:
        return "CRITICAL"
    if dh_dt_cm_per_s > FLOOD_ALERT_DHDT_CM_S:
        return "ALERT"
    if dh_dt_cm_per_s > FLOOD_WATCH_DHDT_CM_S:
        return "WATCH"
    return "NORMAL"


# --------------------------------------------------------------------------
# landslide
# --------------------------------------------------------------------------

def _theta_deep_pct(reading: SensorReading) -> float:
    return (
        reading.soil_4 * DEEP_SOIL_WEIGHTS["soil_4"]
        + reading.soil_5 * DEEP_SOIL_WEIGHTS["soil_5"]
    )


def _tilt_baseline_deg(db: Session, node_id: str) -> float | None:
    """Install-baseline tilt: mean of the first (up to) 200 readings ever received from
    this node. This naturally freezes once 200+ readings exist — 'the first 200 by
    timestamp' never changes after that — so there's no separate lock flag to maintain."""
    rows = (
        db.query(SensorReading)
        .filter(SensorReading.node_id == node_id)
        .order_by(SensorReading.timestamp.asc())
        .limit(TILT_BASELINE_SAMPLE_SIZE)
        .all()
    )
    if not rows:
        return None
    return mean(r.tilt for r in rows)


def _tilt_rate_deg_per_hr(db: Session, node_id: str, now: datetime) -> float | None:
    slope_per_sec = _slope_over_trailing_window(db, node_id, now, lambda r: r.tilt, smooth=False)
    return None if slope_per_sec is None else slope_per_sec * 3600.0


def _rain_duration_seconds(db: Session, node_id: str, now: datetime) -> float:
    """How long rain_state has been continuously True, walking back from the most recent
    reading. 0 if it isn't currently raining."""
    rows = (
        db.query(SensorReading)
        .filter(SensorReading.node_id == node_id)
        .order_by(SensorReading.timestamp.desc())
        .limit(500)  # generous cap — even a multi-hour storm at 30s cadence is a few hundred rows
        .all()
    )
    if not rows or not rows[0].rain_state:
        return 0.0
    earliest_true = rows[0].timestamp
    for row in rows:
        if not row.rain_state:
            break
        earliest_true = row.timestamp
    return (now - earliest_true).total_seconds()


def _landslide_level(
    theta_deep_pct: float,
    tilt_delta_deg: float | None,
    tilt_rate_deg_per_hr: float | None,
    rain_duration_seconds: float,
) -> str:
    rain_min = rain_duration_seconds / 60.0

    if theta_deep_pct > LANDSLIDE_CRITICAL_THETA_PCT and (
        (tilt_delta_deg is not None and tilt_delta_deg > LANDSLIDE_CRITICAL_TILT_DELTA_DEG)
        or (tilt_rate_deg_per_hr is not None and tilt_rate_deg_per_hr > LANDSLIDE_CRITICAL_TILT_RATE_DEG_HR)
    ):
        return "CRITICAL"

    if (
        rain_min > LANDSLIDE_ALERT_RAIN_MIN
        and theta_deep_pct > LANDSLIDE_ALERT_THETA_PCT
        and tilt_delta_deg is not None
        and tilt_delta_deg > LANDSLIDE_ALERT_TILT_DELTA_DEG
    ):
        return "ALERT"

    if rain_min > LANDSLIDE_WATCH_RAIN_MIN and theta_deep_pct > LANDSLIDE_WATCH_THETA_PCT:
        return "WATCH"

    return "NORMAL"


# --------------------------------------------------------------------------
# combined entry point
# --------------------------------------------------------------------------

def _higher(a: str, b: str) -> str:
    return a if LEVELS.index(a) >= LEVELS.index(b) else b


def evaluate_risk(db: Session, node_id: str, now: datetime) -> dict:
    """
    Run both trigger matrices for a node and take the higher level. Every intermediate
    number is returned alongside the verdict — that's the whole "explainable, not a
    black box" point of doing this with rules instead of a model.
    """
    latest = (
        db.query(SensorReading)
        .filter(SensorReading.node_id == node_id)
        .order_by(SensorReading.timestamp.desc())
        .first()
    )
    if latest is None:
        return {
            "level": "NORMAL",
            "flood_level": "NORMAL",
            "landslide_level": "NORMAL",
            "dh_dt_cm_per_s": None,
            "theta_deep_pct": None,
            "tilt_delta_deg": None,
            "tilt_rate_deg_per_hr": None,
            "rain_duration_min": 0.0,
        }

    dh_dt = _flood_dh_dt_cm_per_s(db, node_id, now)
    flood_level = _flood_level(dh_dt, latest.water_level)

    theta_deep = _theta_deep_pct(latest)
    baseline_tilt = _tilt_baseline_deg(db, node_id)
    tilt_delta = abs(latest.tilt - baseline_tilt) if baseline_tilt is not None else None
    tilt_rate = _tilt_rate_deg_per_hr(db, node_id, now)
    rain_duration_seconds = _rain_duration_seconds(db, node_id, now)
    landslide_level = _landslide_level(theta_deep, tilt_delta, tilt_rate, rain_duration_seconds)

    return {
        "level": _higher(flood_level, landslide_level),
        "flood_level": flood_level,
        "landslide_level": landslide_level,
        "dh_dt_cm_per_s": dh_dt,
        "theta_deep_pct": theta_deep,
        "tilt_delta_deg": tilt_delta,
        "tilt_rate_deg_per_hr": tilt_rate,
        "rain_duration_min": rain_duration_seconds / 60.0,
    }
