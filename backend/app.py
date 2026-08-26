"""FastAPI app: /ingest (Node B posts here every cycle) and /status/{village_id} (dashboard polls this)."""

from datetime import datetime

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from db import NodeStatus, SensorReading, get_db, init_db, utcnow
from pipeline import clean_payload, refresh_node_liveness, sweep_dead_nodes

app = FastAPI(title="SIH 26192 — Ridge & Valley Flood/Landslide Backend")


@app.on_event("startup")
def on_startup():
    init_db()


class IngestPayload(BaseModel):
    node_id: str
    timestamp: datetime
    soil_1: float
    soil_2: float
    soil_3: float
    soil_4: float
    soil_5: float
    tilt: float
    rain_state: bool
    water_level: float
    battery: float


class ReadingOut(BaseModel):
    node_id: str
    timestamp: datetime
    soil_1: float
    soil_2: float
    soil_3: float
    soil_4: float
    soil_5: float
    tilt: float
    rain_state: bool
    water_level: float
    battery: float
    flags: str        # notes from the self-healing pipeline, e.g. an IDW fill; "" if none fired
    node_dead: bool    # NodeStatus.is_dead as of this response (>15s since last ingest)

    class Config:
        from_attributes = True


@app.post("/ingest", response_model=ReadingOut)
def ingest(payload: IngestPayload, db: Session = Depends(get_db)):
    """Store one combined payload from Node B, self-heal it, and refresh its liveness record.

    Self-healing (Phase 3): Z-score outlier rejection per channel, IDW fill for a dead
    soil sensor from its depth neighbours, and a sweep marking any node DEAD if it's
    gone silent past the 15s offline threshold. Risk scoring (Phase 4) is not wired
    in yet — this endpoint stores the cleaned reading.
    """
    now = utcnow()
    cleaned = clean_payload(db, payload.model_dump())
    flags = cleaned.pop("_flags")

    reading = SensorReading(**cleaned, flags="; ".join(flags))
    db.add(reading)

    refresh_node_liveness(db, payload.node_id, now)
    sweep_dead_nodes(db, now)  # covers every known node, not just this one

    db.commit()
    db.refresh(reading)

    status = db.get(NodeStatus, payload.node_id)
    return _to_reading_out(reading, status)


@app.get("/status/{village_id}", response_model=ReadingOut)
def get_status(village_id: str, db: Session = Depends(get_db)):
    """Latest stored reading for a node. `village_id` == the node_id Node B sends."""
    sweep_dead_nodes(db, utcnow())
    db.commit()

    reading = (
        db.query(SensorReading)
        .filter(SensorReading.node_id == village_id)
        .order_by(SensorReading.timestamp.desc())
        .first()
    )
    if reading is None:
        raise HTTPException(status_code=404, detail=f"No readings for node '{village_id}'")

    status = db.get(NodeStatus, village_id)
    return _to_reading_out(reading, status)


def _to_reading_out(reading: SensorReading, status: NodeStatus) -> ReadingOut:
    fields = {
        col: getattr(reading, col)
        for col in ReadingOut.model_fields
        if col != "node_dead" and hasattr(reading, col)
    }
    fields["node_dead"] = bool(status.is_dead) if status else False
    return ReadingOut(**fields)
