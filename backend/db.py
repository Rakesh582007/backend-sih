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
    """One combined payload from Node B: its own sensors + Node A's relayed LoRa packet."""

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

    tilt = Column(Float, nullable=False)
    rain_state = Column(Boolean, nullable=False)

    # Node B's own ultrasonic sensor
    water_level = Column(Float, nullable=False)
    battery = Column(Float, nullable=False)

    # Comma-separated notes from the self-healing pipeline (Phase 3), e.g. an IDW fill
    # or a flagged-but-not-corrected tilt/water_level spike. Empty string if nothing fired.
    flags = Column(Text, nullable=False, default="")

    received_at = Column(DateTime, default=utcnow)


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
