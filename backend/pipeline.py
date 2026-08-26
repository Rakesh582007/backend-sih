"""Self-healing pipeline (Phase 3): Z-score outlier rejection, dead-node marking, IDW fill.

Design note on WHICH channels get auto-corrected vs. just flagged:

- The 5 soil sensors are spatially redundant (they sample a continuous wetting
  front at known depths), so when one looks like a wild outlier we can safely
  estimate a replacement from its depth neighbours via IDW and keep the
  pipeline running.
- tilt and water_level are the exact channels the risk engine (Phase 4) uses
  to catch a real landslide/flood precursor — a sudden jump IS the signal,
  not noise. Silently overwriting a spike here could hide the one reading
  that mattered. So these are flagged as statistically unusual but stored
  as-is; Phase 4 / a human on the dashboard decides what to do with a flag.
- battery is a monotonically-draining trend, not a channel that "spikes", so
  it isn't Z-score checked at all here.
"""

from datetime import datetime, timedelta
from statistics import mean, pstdev

from sqlalchemy.orm import Session

from db import NodeStatus, SensorReading

# --- config ---
OFFLINE_THRESHOLD_SECONDS = 15  # per manual: mark a node DEAD if silent this long
Z_SCORE_HISTORY_WINDOW = 20     # how many past (already-cleaned) readings form the baseline
Z_SCORE_MIN_HISTORY = 5         # don't reject anything until the baseline has this many samples
Z_SCORE_THRESHOLD = 3.0         # standard "wild outlier" cutoff

SOIL_CHANNELS = ["soil_1", "soil_2", "soil_3", "soil_4", "soil_5"]

# Install depths (cm) — used for inverse-distance weighting between channels
SOIL_DEPTHS_CM = {
    "soil_1": 10,
    "soil_2": 30,
    "soil_3": 60,
    "soil_4": 100,
    "soil_5": 150,
}

# Safety-relevant channels: outliers are flagged, never silently overwritten
FLAG_ONLY_CHANNELS = ["tilt", "water_level"]


def _history(db: Session, node_id: str, channel: str, limit: int = Z_SCORE_HISTORY_WINDOW):
    """Most-recent-first list of past values for one node+channel, used as the Z-score baseline."""
    rows = (
        db.query(SensorReading)
        .filter(SensorReading.node_id == node_id)
        .order_by(SensorReading.timestamp.desc())
        .limit(limit)
        .all()
    )
    return [getattr(r, channel) for r in rows]


def _is_outlier(value: float, history: list) -> bool:
    """Z-score check against `history`. Never rejects until there's enough history to trust."""
    if len(history) < Z_SCORE_MIN_HISTORY:
        return False
    mu = mean(history)
    sigma = pstdev(history)
    if sigma == 0:
        # No variance to compute a z-score against (e.g. a perfectly flat baseline).
        # Don't reject — treating every natural reading as an outlier here would be
        # a false-positive storm, not self-healing.
        return False
    z = abs((value - mu) / sigma)
    return z > Z_SCORE_THRESHOLD


def _idw_estimate(known_values: dict, target_channel: str, power: float = 2.0):
    """Inverse-distance-weighted estimate for one soil channel from its healthy depth neighbours."""
    target_depth = SOIL_DEPTHS_CM[target_channel]
    weighted_sum = 0.0
    weight_total = 0.0
    for channel, value in known_values.items():
        distance = abs(SOIL_DEPTHS_CM[channel] - target_depth)
        if distance == 0:
            return value
        weight = 1.0 / (distance ** power)
        weighted_sum += weight * value
        weight_total += weight
    if weight_total == 0:
        return None
    return weighted_sum / weight_total


def clean_payload(db: Session, payload: dict) -> dict:
    """
    Run the self-healing checks over one incoming (already schema-validated) payload.
    Returns a copy of `payload` with soil outliers replaced by their IDW estimate and
    an added "_flags" key (list[str]) describing everything the pipeline did.
    """
    node_id = payload["node_id"]
    cleaned = dict(payload)
    flags = []

    # --- Z-score outlier rejection, soil channels ---
    dead_soil_channels = [
        channel
        for channel in SOIL_CHANNELS
        if _is_outlier(cleaned[channel], _history(db, node_id, channel))
    ]

    # --- IDW fill for any soil channel flagged dead, from its still-healthy neighbours ---
    if dead_soil_channels:
        healthy = {ch: cleaned[ch] for ch in SOIL_CHANNELS if ch not in dead_soil_channels}
        for channel in dead_soil_channels:
            estimate = _idw_estimate(healthy, channel)
            if estimate is None:
                # All 5 soil sensors flagged dead at once — nothing left to interpolate from.
                flags.append(f"{channel}: rejected outlier, no healthy neighbours to interpolate from")
                continue
            flags.append(f"{channel}: rejected outlier {cleaned[channel]:.2f}, IDW-estimated {estimate:.2f}")
            cleaned[channel] = estimate

    # --- Z-score check, safety-relevant channels: flag only, never overwrite ---
    for channel in FLAG_ONLY_CHANNELS:
        if _is_outlier(cleaned[channel], _history(db, node_id, channel)):
            flags.append(f"{channel}: statistical outlier ({cleaned[channel]}) — kept as-is, not auto-corrected")

    cleaned["_flags"] = flags
    return cleaned


def refresh_node_liveness(db: Session, node_id: str, now: datetime) -> NodeStatus:
    """Upsert NodeStatus for a node that just sent data: bump last_seen, clear is_dead."""
    status = db.get(NodeStatus, node_id)
    if status is None:
        status = NodeStatus(node_id=node_id, last_seen=now, is_dead=False)
        db.add(status)
    else:
        status.last_seen = now
        status.is_dead = False
    return status


def sweep_dead_nodes(db: Session, now: datetime) -> list:
    """
    Mark any node DEAD if it's gone silent longer than OFFLINE_THRESHOLD_SECONDS.
    No background scheduler in this hackathon build — call this on every /ingest and
    /status hit so the flag stays accurate as long as *something* is polling the API.
    Returns the node_ids newly marked dead.
    """
    cutoff = now - timedelta(seconds=OFFLINE_THRESHOLD_SECONDS)
    newly_dead = []
    for status in db.query(NodeStatus).filter(NodeStatus.is_dead.is_(False)).all():
        if status.last_seen < cutoff:
            status.is_dead = True
            newly_dead.append(status.node_id)
    return newly_dead
