"""FastAPI app: /ingest (Node B posts here every cycle) and /status/{village_id} (dashboard polls this)."""

from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from db import NodeStatus, SensorReading, get_db, init_db

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

    class Config:
        from_attributes = True


@app.post("/ingest", response_model=ReadingOut)
def ingest(payload: IngestPayload, db: Session = Depends(get_db)):
    """Store one combined payload from Node B and refresh its liveness record.

    NOTE: this is the raw store step (Phase 2). Outlier rejection / dead-node
    handling / IDW fill (Phase 3) and risk scoring (Phase 4) are not wired in
    here yet — this endpoint just persists what it's given.
    """
    reading = SensorReading(**payload.model_dump())
    db.add(reading)

    now = datetime.now(timezone.utc)
    status = db.get(NodeStatus, payload.node_id)
    if status is None:
        status = NodeStatus(node_id=payload.node_id, last_seen=now, is_dead=False)
        db.add(status)
    else:
        status.last_seen = now
        status.is_dead = False

    db.commit()
    db.refresh(reading)
    return reading


@app.get("/status/{village_id}", response_model=ReadingOut)
def get_status(village_id: str, db: Session = Depends(get_db)):
    """Latest stored reading for a node. `village_id` == the node_id Node B sends."""
    reading = (
        db.query(SensorReading)
        .filter(SensorReading.node_id == village_id)
        .order_by(SensorReading.timestamp.desc())
        .first()
    )
    if reading is None:
        raise HTTPException(status_code=404, detail=f"No readings for node '{village_id}'")
    return reading
