"""
GiriKavach — FastAPI Edge Server (merged)
============================================

Runs on the village gateway laptop. NOT in the cloud — the whole point
is that this works with no internet. "Backend" here means "the machine
in the Panchayat office", not "a server in a datacentre".

THIS FILE IS THE RESULT OF MERGING TWO INDEPENDENTLY-BUILT BACKENDS
  - app.py + db.py + pipeline.py + risk_engine.py (Rakesh's): SQLAlchemy
    persistence, self-healing (Z-score outlier rejection + IDW fill),
    the theta_deep_pct soil-saturation formula.
  - The original main.py + false_positive_filter.py + cap_generator.py +
    llm_narrator.py + open_meteo.py (this side): XGBoost inference, the
    false-positive filter (hysteresis + safety net), CAP 1.2, local LLM
    narration.

STRICT HIERARCHY — THE ONE RULE THIS MERGE MUST NEVER BREAK
FalsePositiveFilter.evaluate() is the ONLY thing that sets
should_sound_siren. Nothing before it (self-healing, theta_deep_pct,
XGBoost, Open-Meteo pre-arm, Phase 5 susceptibility) may trigger or
veto an alert directly — they can only feed it better-cleaned inputs or
adjust its sensitivity CONFIG (consecutive_to_escalate). Nothing after
it (CAP generation, LLM narration, DB persistence) can change its
verdict — they only describe or store it. risk_engine.py's OWN
trigger-matrix decision function (evaluate_risk()) is intentionally NOT
imported here — it was Rakesh's original rule-based decision path,
superseded by the filter for the merged decision path. Only its pure
theta_deep_pct() helper is reused (see risk_engine.py).

PIPELINE (per reading)
    /ingest
      -> pipeline.clean_payload()      self-healing FIRST: Z-score
                                        reject + IDW-fill soil channels,
                                        flag (never silently overwrite)
                                        slope/stream-depth spikes
      -> risk_engine.theta_deep_pct()  deep_soil_moisture_pct, from the
                                        CLEANED soil_4/soil_5
      -> XGBoost (flood + landslide)   the only ML in the live decision
      -> FalsePositiveFilter.evaluate() THE decision. Authoritative.
      -> CAP 1.2 + LLM narration        describe the decision, never
                                         change it
      -> BackgroundTasks: persist reading + decision via Rakesh's
                                         SQLAlchemy models — fire and
                                         forget, runs AFTER the response
                                         is already sent

IMPORTANT — this is the ONLINE path. If this laptop is off, unplugged,
or crashed, the ESP32 still runs its own TFLM model and still fires the
buzzer. Nothing here is load-bearing for the physical alert. This layer
adds the dashboard, the CAP payload, persistence, and the higher-
accuracy model.

RUN:
    uvicorn main:app --reload --port 8000
    open http://127.0.0.1:8000/docs   (interactive API explorer)
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path

import joblib
import pandas as pd
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from db import Decision, SessionLocal, SensorReading as DbSensorReading, get_db, init_db, utcnow
from pipeline import clean_payload, refresh_node_liveness, sweep_dead_nodes
from risk_engine import theta_deep_pct
from cap_generator import AlertArea, build_cap_alert
from false_positive_filter import (
    FalsePositiveFilter,
    FilterConfig,
    RiskLevel,
    SensorReading as FilterReading,   # name clash with db.SensorReading — alias
)
from open_meteo import POLL_INTERVAL_MINUTES, fetch_prearm
from susceptibility import load_susceptibility

# Alert narration. These modules are optional by design: if either is
# missing the server still ingests, still decides, still fires the siren
# and still emits CAP. Only the human-readable wording degrades.
try:
    from llm_narrator import NarrationRequest, narrate, situation_report
    NARRATION_AVAILABLE = True
except ImportError:
    NARRATION_AVAILABLE = False

IST = timezone(timedelta(hours=5, minutes=30))

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"

# Deployment site — set per installation.
SITE_AREA = AlertArea(
    description="Monitored catchment (configure per deployment site)",
    latitude=11.4654,
    longitude=76.1358,
    radius_km=5.0,
)

state: dict = {
    "flood_model": None,
    "landslide_model": None,
    "filter": None,
    "last_decision": None,
    "last_reading": None,
    "last_updated": None,
    "history": [],       # recent decisions, for the dashboard
    "prearm": None,       # latest PreArmState from Open-Meteo
}


async def _prearm_loop():
    """Poll Open-Meteo on a timer and adjust filter sensitivity.

    Runs forever in the background. Every failure path inside
    fetch_prearm() returns a safe state, so a dead internet connection
    simply leaves the filter at its normal settings — which is exactly
    what we want, since no-internet is the normal condition during the
    storm this system exists for.
    """
    while True:
        try:
            site = SITE_AREA
            prearm = fetch_prearm(site.latitude, site.longitude)
            state["prearm"] = prearm

            # Apply sensitivity change to the LIVE filter. Note this only
            # ever touches how many consecutive readings confirm an
            # escalation — it cannot create an escalation on its own.
            filt = state.get("filter")
            if filt is not None:
                filt.cfg.consecutive_to_escalate = prearm.consecutive_to_escalate

            print(f"[prearm] active={prearm.active} — {prearm.reason}")
        except Exception as e:
            # Never let the background task die; a crashed poller must
            # not take down ingestion.
            print(f"[prearm] poll error (ignored): {type(e).__name__}: {e}")

        await asyncio.sleep(POLL_INTERVAL_MINUTES * 60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once at startup rather than per-request."""
    try:
        state["flood_model"] = joblib.load(MODEL_DIR / "flood_model.pkl")
        state["landslide_model"] = joblib.load(MODEL_DIR / "landslide_model.pkl")
        print(f"[startup] models loaded from {MODEL_DIR}")
    except FileNotFoundError as e:
        # Fail loudly but keep serving — /health will report the problem
        # rather than the whole service silently refusing connections.
        print(f"[startup] WARNING: model load failed: {e}")
    state["filter"] = FalsePositiveFilter(FilterConfig())
    print(f"[startup] narration available: {NARRATION_AVAILABLE}")

    init_db()

    task = asyncio.create_task(_prearm_loop())
    yield
    task.cancel()


app = FastAPI(
    title="GiriKavach Edge Server",
    description="Offline-first dual-hazard early warning gateway (merged backend)",
    version="2.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------
# Request/response schemas
# ---------------------------------------------------------------------
class TelemetryIn(BaseModel):
    """Merged Node1 (LoRa, 5-depth soil profile) + Node2 (local) sensor
    sample. Field names match schema.json's reconciled contract.

    Defaults let a partial packet through rather than 422-ing during a
    real event — a missing optional field should never take the gateway
    offline.
    """
    node_id: str

    soil_1: float = Field(0.0, ge=0, le=100)
    soil_2: float = Field(0.0, ge=0, le=100)
    soil_3: float = Field(0.0, ge=0, le=100)
    soil_4: float = Field(0.0, ge=0, le=100)
    soil_5: float = Field(0.0, ge=0, le=100)

    rain_intensity_mm_hr: float = Field(0.0, ge=0)
    rain_duration_min: float = Field(0.0, ge=0)

    slope_pitch_deg: float = Field(0.0)
    pitch_rate_deg_min: float = Field(0.0)
    vibration_intensity: float = Field(0.0, ge=0)
    gyro_angular_vel_deg_s: float = Field(0.0, ge=0)

    stream_depth_cm: float = Field(0.0)
    rate_of_rise_cm_min: float = Field(0.0)

    node1_link_fresh: bool = True
    battery_voltage: float = Field(0.0)
    rssi_dbm: int | None = None


class DecisionOut(BaseModel):
    timestamp: str
    node_id: str
    risk_level: str
    flood_probability: float
    landslide_probability: float
    theta_deep_pct: float
    should_sound_siren: bool
    safety_net_triggered: bool
    reasons: list[str]
    suppressed: list[str]
    sensor_health_warnings: list[str]
    alert_text: str | None = None
    alert_text_source: str | None = None
    prearm_active: bool = False


# ---------------------------------------------------------------------
def _predict(features: dict) -> tuple[float, float]:
    """Run both XGBoost models against an already-assembled feature dict.
    Returns (flood_pct, landslide_pct)."""
    fm, lm = state["flood_model"], state["landslide_model"]
    if fm is None or lm is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded — run train_xgboost_models.py first",
        )
    flood_cols = ["rain_intensity_mm_hr", "rain_duration_min", "deep_soil_moisture_pct",
                  "stream_depth_cm", "rate_of_rise_cm_min"]
    slide_cols = ["deep_soil_moisture_pct", "slope_pitch_deg", "pitch_rate_deg_min",
                  "vibration_intensity", "gyro_angular_vel_deg_s"]
    flood_df = pd.DataFrame([{k: features[k] for k in flood_cols}])
    slide_df = pd.DataFrame([{k: features[k] for k in slide_cols}])
    flood_p = float(fm.predict(flood_df)[0])
    slide_p = float(lm.predict(slide_df)[0])
    # Models are regressors — clamp to a valid probability range.
    return max(0.0, min(100.0, flood_p)), max(0.0, min(100.0, slide_p))


def _persist_reading_and_decision(
    node_id: str,
    now: datetime,
    cleaned: dict,
    flags: list[str],
    decision,
    theta: float,
) -> None:
    """Runs AFTER the response has already been sent (FastAPI
    BackgroundTasks semantics) — this can NEVER delay should_sound_siren
    reaching the caller, by construction, not just by convention.

    Opens its own DB session rather than reusing the request-scoped one
    from Depends(get_db): Starlette tears down request dependencies
    (closing that session) before background tasks run, so reusing it
    here would be a use-after-close bug, not just bad style.

    Any failure here is logged and swallowed, never raised — the alert
    already went out; a failed audit-log write must not look like a
    failed alert.
    """
    db = SessionLocal()
    try:
        reading = DbSensorReading(
            node_id=node_id,
            timestamp=now,
            soil_1=cleaned["soil_1"], soil_2=cleaned["soil_2"], soil_3=cleaned["soil_3"],
            soil_4=cleaned["soil_4"], soil_5=cleaned["soil_5"],
            rain_intensity_mm_hr=cleaned["rain_intensity_mm_hr"],
            rain_duration_min=cleaned["rain_duration_min"],
            slope_pitch_deg=cleaned["slope_pitch_deg"],
            pitch_rate_deg_min=cleaned["pitch_rate_deg_min"],
            vibration_intensity=cleaned["vibration_intensity"],
            gyro_angular_vel_deg_s=cleaned["gyro_angular_vel_deg_s"],
            stream_depth_cm=cleaned["stream_depth_cm"],
            rate_of_rise_cm_min=cleaned["rate_of_rise_cm_min"],
            node1_link_fresh=cleaned["node1_link_fresh"],
            battery_voltage=cleaned["battery_voltage"],
            rssi_dbm=cleaned.get("rssi_dbm"),
            flags="; ".join(flags),
        )
        db.add(reading)

        refresh_node_liveness(db, node_id, now)
        sweep_dead_nodes(db, now)

        db.add(Decision(
            node_id=node_id,
            timestamp=now,
            risk_level=decision.level.name,
            flood_probability=decision.flood_probability,
            landslide_probability=decision.landslide_probability,
            theta_deep_pct=theta,
            should_sound_siren=decision.should_sound_siren,
            safety_net_triggered=decision.safety_net_triggered,
            reasons="; ".join(decision.reasons),
            suppressed="; ".join(decision.suppressed),
            sensor_health_warnings="; ".join(decision.sensor_health_warnings),
        ))

        db.commit()
    except Exception as e:
        print(f"[persist] FAILED (alert already sent, unaffected): {type(e).__name__}: {e}")
    finally:
        db.close()


@app.get("/health")
def health():
    """Is the gateway alive, and are the models actually loaded?"""
    prearm = state.get("prearm")
    return {
        "status": "ok",
        "models_loaded": state["flood_model"] is not None and state["landslide_model"] is not None,
        "narration_available": NARRATION_AVAILABLE,
        "prearm_active": bool(prearm and prearm.active),
        "last_updated": state["last_updated"],
        "note": "Physical siren does NOT depend on this service — ESP32 runs TFLM independently",
    }


@app.post("/ingest", response_model=DecisionOut)
def ingest(
    reading: TelemetryIn,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    language: str = "en",
):
    """Main entry point. ESP32 serial bridge / Node B POSTs merged telemetry here.

    `db` (the request-scoped session) is used ONLY for the read-only
    history lookups self-healing needs (Z-score baselines) — every
    WRITE happens in the background task, on its own session, after the
    response is already on the wire.
    """
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)  # matches db.utcnow()'s naive-UTC convention

    payload = reading.model_dump()

    # --- Self-healing FIRST (step 6): Z-score reject + IDW-fill soil
    # channels; flag (never silently overwrite) slope/stream-depth
    # spikes. Everything downstream — theta_deep_pct, XGBoost, the
    # filter — sees only CLEANED values.
    cleaned = clean_payload(db, payload)
    flags = cleaned.pop("_flags")

    # --- The one reconciled soil-saturation formula, from the CLEANED
    # (post-imputation) soil_4/soil_5.
    theta = theta_deep_pct(cleaned["soil_4"], cleaned["soil_5"])

    # --- XGBoost: the only ML in the live decision path ---
    features = {
        "rain_intensity_mm_hr": cleaned["rain_intensity_mm_hr"],
        "rain_duration_min": cleaned["rain_duration_min"],
        "deep_soil_moisture_pct": theta,
        "stream_depth_cm": cleaned["stream_depth_cm"],
        "rate_of_rise_cm_min": cleaned["rate_of_rise_cm_min"],
        "slope_pitch_deg": cleaned["slope_pitch_deg"],
        "pitch_rate_deg_min": cleaned["pitch_rate_deg_min"],
        "vibration_intensity": cleaned["vibration_intensity"],
        "gyro_angular_vel_deg_s": cleaned["gyro_angular_vel_deg_s"],
    }
    flood_p, slide_p = _predict(features)

    # --- Phase 5 susceptibility (currently always neutral — see
    # susceptibility.py): may only adjust filter sensitivity, exactly
    # like Open-Meteo pre-arm above. Never touches should_sound_siren.
    susc = load_susceptibility(reading.node_id)
    if susc.consecutive_to_escalate_override is not None:
        state["filter"].cfg.consecutive_to_escalate = susc.consecutive_to_escalate_override

    # --- FalsePositiveFilter: THE ONLY thing that sets
    # should_sound_siren. Nothing before this point, and nothing after
    # it, may veto or trigger an alert.
    filter_reading = FilterReading(
        rain_intensity_mm_hr=cleaned["rain_intensity_mm_hr"],
        rain_duration_min=cleaned["rain_duration_min"],
        deep_soil_moisture_pct=theta,
        slope_pitch_deg=cleaned["slope_pitch_deg"],
        pitch_rate_deg_min=cleaned["pitch_rate_deg_min"],
        vibration_intensity=cleaned["vibration_intensity"],
        gyro_angular_vel_deg_s=cleaned["gyro_angular_vel_deg_s"],
        stream_depth_cm=cleaned["stream_depth_cm"],
        rate_of_rise_cm_min=cleaned["rate_of_rise_cm_min"],
        node1_link_fresh=cleaned["node1_link_fresh"],
    )
    decision = state["filter"].evaluate(filter_reading, flood_p, slide_p)

    # Self-healing's own notes (e.g. an IDW fill) folded into the SAME
    # honesty channel as the filter's sensor-health findings — step 6:
    # imputed/estimated values must be visible here, not silently
    # indistinguishable from a real reading.
    sensor_health_warnings = list(decision.sensor_health_warnings) + list(flags)

    # Citizen-facing wording. Generated from reviewed templates, never
    # free-form — see llm_narrator.py for why. Failure here must not
    # break ingestion, so it is wrapped.
    alert_text, alert_source = None, None
    if NARRATION_AVAILABLE and decision.level >= RiskLevel.WATCH:
        try:
            hazard = (
                "landslide"
                if decision.landslide_probability >= decision.flood_probability
                else "flood"
            )
            result = narrate(
                NarrationRequest(
                    risk_level=decision.level.name,
                    hazard=hazard,
                    probability=max(decision.flood_probability, decision.landslide_probability),
                    place_name=SITE_AREA.description,
                    reasons=decision.reasons[:4],
                    language=language,
                    is_exercise=True,   # flip only for a commissioned deployment
                )
            )
            alert_text = result.get("text")
            alert_source = result.get("source")
        except Exception as e:
            print(f"[narrate] failed (ignored): {type(e).__name__}: {e}")

    prearm = state.get("prearm")
    now_ist = datetime.now(IST).isoformat(timespec="seconds")
    out = DecisionOut(
        timestamp=now_ist,
        node_id=reading.node_id,
        risk_level=decision.level.name,
        flood_probability=round(decision.flood_probability, 1),
        landslide_probability=round(decision.landslide_probability, 1),
        theta_deep_pct=round(theta, 1),
        should_sound_siren=decision.should_sound_siren,
        safety_net_triggered=decision.safety_net_triggered,
        reasons=decision.reasons,
        suppressed=decision.suppressed,
        sensor_health_warnings=sensor_health_warnings,
        alert_text=alert_text,
        alert_text_source=alert_source,
        prearm_active=bool(prearm and prearm.active),
    )

    # --- Persist as a background task: fire-and-forget, executes AFTER
    # this response is already sent. Cannot block or delay
    # should_sound_siren reaching the caller.
    background_tasks.add_task(
        _persist_reading_and_decision, reading.node_id, now_utc, cleaned, flags, decision, theta,
    )

    state["last_decision"] = decision
    state["last_reading"] = filter_reading
    state["last_updated"] = now_ist
    state["history"].append(out.model_dump())
    state["history"] = state["history"][-500:]   # bounded — this runs for months
    return out


@app.get("/status", response_model=DecisionOut | None)
def status():
    """Latest decision — what the dashboard polls."""
    if not state["history"]:
        return None
    return state["history"][-1]


@app.get("/history")
def history(limit: int = 100):
    """Recent decisions for dashboard charts."""
    return {"count": len(state["history"][-limit:]), "items": state["history"][-limit:]}


@app.get("/alert/cap", response_class=PlainTextResponse)
def cap_alert(status_value: str = "Exercise"):
    """CAP 1.2 XML for the current risk state.

    status_value defaults to "Exercise" — see cap_generator.py for why.
    Only pass "Actual" from a real, commissioned deployment.
    """
    decision = state["last_decision"]
    reading = state["last_reading"]
    if decision is None or reading is None:
        raise HTTPException(status_code=404, detail="No telemetry received yet")

    if decision.landslide_probability >= decision.flood_probability:
        hazard, probability = "landslide", decision.landslide_probability
    else:
        hazard, probability = "flood", decision.flood_probability

    try:
        return build_cap_alert(
            hazard=hazard,
            risk_level=decision.level.name,
            probability=probability,
            area=SITE_AREA,
            status=status_value,
            reasons=decision.reasons[:4],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/filter/reset")
def reset_filter():
    """Clear filter history (hysteresis state, medians). Demo/testing aid."""
    state["filter"] = FalsePositiveFilter(FilterConfig())
    # Re-apply any active pre-arm to the fresh filter.
    prearm = state.get("prearm")
    if prearm is not None:
        state["filter"].cfg.consecutive_to_escalate = prearm.consecutive_to_escalate
    return {"status": "filter state reset"}


@app.get("/prearm")
def prearm_status():
    """Current Open-Meteo pre-arm state.

    active=False is always safe — it means the filter is at its normal
    (less sensitive) settings, which is where it sits whenever the
    forecast is calm OR the internet is unreachable.
    """
    prearm = state.get("prearm")
    if prearm is None:
        return {
            "active": False,
            "reason": "not polled yet (first poll runs at startup)",
            "consecutive_to_escalate": 3,
        }
    return {
        "active": prearm.active,
        "reason": prearm.reason,
        "peak_forecast_mm_hr": prearm.peak_forecast_mm_hr,
        "hours_until_peak": prearm.hours_until_peak,
        "fetched_at": prearm.fetched_at,
        "consecutive_to_escalate": prearm.consecutive_to_escalate,
        "source": prearm.source,
    }


@app.get("/sitrep", response_class=PlainTextResponse)
def sitrep():
    """Plain-language situation report for the ops room (English only).

    This is the ONE place an LLM writes prose in this system, and it is
    internal — read by an operator who can sanity-check it, never sent
    to a citizen. Falls back to raw facts if the model is unavailable.
    """
    if not NARRATION_AVAILABLE:
        return "Narration module not available."
    if not state["history"]:
        return "No telemetry received yet."
    try:
        return situation_report(state["history"][-30:])
    except Exception as e:
        return f"Situation report unavailable ({type(e).__name__}). Latest: {state['history'][-1]}"


@app.get("/alert/text")
def alert_text(language: str = "en"):
    """Citizen alert wording for the current state, in a given language.

    Text comes from reviewed templates — see llm_narrator.py / alert_templates.py.
    """
    if not NARRATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Narration module not available")
    decision = state["last_decision"]
    if decision is None:
        raise HTTPException(status_code=404, detail="No telemetry received yet")

    hazard = (
        "landslide"
        if decision.landslide_probability >= decision.flood_probability
        else "flood"
    )
    result = narrate(
        NarrationRequest(
            risk_level=decision.level.name,
            hazard=hazard,
            probability=max(decision.flood_probability, decision.landslide_probability),
            place_name=SITE_AREA.description,
            reasons=decision.reasons[:4],
            language=language,
            is_exercise=True,
        )
    )
    return result
