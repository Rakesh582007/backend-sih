"""SQLAlchemy models + engine setup: SensorReading, NodeStatus."""

from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker


def utcnow() -> datetime:
    """Naive UTC 'now'. SQLite's DATETIME column drops tzinfo on round-trip, so every
    datetime this app produces or compares is kept naive-but-UTC-by-convention to avoid
    aware/naive comparison bugs between freshly-created and stored values."""
    return datetime.now(timezone.utc).replace(tzinfo=None)

DATABASE_URL = "sqlite:///./sih.db"

# check_same_thread=False is needed because FastAPI can hand a request to a
# different thread than the one that created the connection; SQLite is fine
# with this as long as we don't share a session across threads (we don't —
# get_db() hands out one session per request).
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class SensorReading(Base):
    """One combined payload from Node B: its own sensors + Node A's relayed LoRa packet.

    Column names updated as part of the backend merge to match schema.json
    (the reconciled data contract both sides agreed on) rather than the
    original MVP names: tilt -> slope_pitch_deg, water_level ->
    stream_depth_cm, battery -> battery_voltage, and the old boolean
    rain_state replaced by the richer rain_intensity_mm_hr/rain_duration_min
    the XGBoost models actually need. The self-healing pipeline (pipeline.py)
    is completely unchanged in its algorithm — only the column names it
    reads via getattr() moved to match.
    """

    __tablename__ = "sensor_readings"

    id = Column(Integer, primary_key=True, index=True)
    node_id = Column(String, index=True, nullable=False)
    timestamp = Column(DateTime, nullable=False)

    # 5 depth sensors from Node A (ridge), relayed over LoRa
    soil_1 = Column(Float, nullable=False)
    soil_2 = Column(Float, nullable=False)
    soil_3 = Column(Float, nullable=False)
    soil_4 = Column(Float, nullable=False)
    soil_5 = Column(Float, nullable=False)

    rain_intensity_mm_hr = Column(Float, nullable=False, default=0.0)
    rain_duration_min = Column(Float, nullable=False, default=0.0)

    slope_pitch_deg = Column(Float, nullable=False)         # was `tilt`
    pitch_rate_deg_min = Column(Float, nullable=False, default=0.0)
    vibration_intensity = Column(Float, nullable=False, default=0.0)
    gyro_angular_vel_deg_s = Column(Float, nullable=False, default=0.0)

    # Node B's own ultrasonic sensor
    stream_depth_cm = Column(Float, nullable=False)         # was `water_level`
    rate_of_rise_cm_min = Column(Float, nullable=False, default=0.0)

    node1_link_fresh = Column(Boolean, nullable=False, default=True)
    battery_voltage = Column(Float, nullable=False, default=0.0)   # was `battery`
    rssi_dbm = Column(Integer, nullable=True)

    # Comma-separated notes from the self-healing pipeline (Phase 3), e.g. an IDW fill
    # or a flagged-but-not-corrected slope/stream-depth spike. Empty string if nothing fired.
    flags = Column(Text, nullable=False, default="")

    received_at = Column(DateTime, default=utcnow)


class Decision(Base):
    """The FalsePositiveFilter's verdict for one reading — persisted
    separately from SensorReading because it's a DIFFERENT kind of fact:
    SensorReading is what was measured, Decision is what was concluded
    from it. Rakesh's original app.py computed risk on-demand in GET
    /status rather than storing it; the merge stores it because step 4
    of the merge plan requires it be persisted, and because "what did we
    decide and why, at the time" is exactly the audit trail a life-safety
    system needs — recomputing it later from raw readings wouldn't
    reproduce the filter's stateful hysteresis at that moment.

    should_sound_siren here is a straight copy of what
    FalsePositiveFilter.evaluate() returned — this table stores that
    verdict, it never recomputes or second-guesses it.
    """

    __tablename__ = "decisions"

    id = Column(Integer, primary_key=True, index=True)
    node_id = Column(String, index=True, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=utcnow)

    risk_level = Column(String, nullable=False)
    flood_probability = Column(Float, nullable=False)
    landslide_probability = Column(Float, nullable=False)
    theta_deep_pct = Column(Float, nullable=True)

    should_sound_siren = Column(Boolean, nullable=False)
    safety_net_triggered = Column(Boolean, nullable=False)

    # Semicolon-joined, matching SensorReading.flags' convention.
    reasons = Column(Text, nullable=False, default="")
    suppressed = Column(Text, nullable=False, default="")
    sensor_health_warnings = Column(Text, nullable=False, default="")


class NodeStatus(Base):
    """Tracks liveness per node so the pipeline can mark a node DEAD after the offline threshold."""

    __tablename__ = "node_status"

    node_id = Column(String, primary_key=True, index=True)
    last_seen = Column(DateTime, nullable=False)
    is_dead = Column(Boolean, default=False, nullable=False)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
