"""SQLAlchemy models + engine setup: SensorReading, NodeStatus."""

from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

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

    received_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


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
