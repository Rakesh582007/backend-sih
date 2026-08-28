"""Shared contract loader for GiriKavach ML + dashboard.

Single source of truth: ``../schema.json`` (repo root, one level above
``backend/``). Both ``backend/ml/train.py`` and
``backend/dashboard/streamlit_app.py`` import THIS module instead of
hard-coding feature order, tier strings, or thresholds a second time.

Golden rule (see CLAUDE.md): if this module and schema.json ever disagree,
schema.json wins and the code must fail loudly, not silently pick a side.
"""

from __future__ import annotations

import json
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

# This file lives at backend/ml/contract.py; the schema is at the repo root,
# two directories up (backend/ml -> backend -> repo root).
SCHEMA_PATH = Path(__file__).resolve().parent.parent.parent / "schema.json"

# The schema version this module was written against. On import we check the
# live file still matches and warn loudly (not silently) if it has moved on
# without this loader being reviewed against the new version.
EXPECTED_SCHEMA_VERSION = "1.0.0"


@lru_cache(maxsize=1)
def load_schema() -> dict[str, Any]:
    """Parse and cache ../schema.json. Raises loudly if it's missing —
    never falls back to an invented/default schema."""
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(
            f"Frozen contract not found at {SCHEMA_PATH}. Per CLAUDE.md, do not "
            f"invent a path or a schema — stop and ask before working around this."
        )
    with SCHEMA_PATH.open(encoding="utf-8") as f:
        schema = json.load(f)

    version = schema.get("version")
    if version != EXPECTED_SCHEMA_VERSION:
        warnings.warn(
            f"schema.json version is {version!r}, but contract.py was written "
            f"against {EXPECTED_SCHEMA_VERSION!r}. The contract may have moved "
            f"without this loader being updated to match — verify feature order, "
            f"tiers, and thresholds below by hand before trusting them.",
            stacklevel=2,
        )
    return schema


def flood_features() -> list[str]:
    """Ordered feature names for the flood model. Order is load-bearing —
    read from schema.json, never retyped from memory."""
    return list(load_schema()["ml_feature_vectors"]["flood_model"])


def landslide_features() -> list[str]:
    """Ordered feature names for the landslide model. Order is load-bearing —
    read from schema.json, never retyped from memory."""
    return list(load_schema()["ml_feature_vectors"]["landslide_model"])


def assert_feature_order(model_name: str, columns: list[str]) -> None:
    """Fail loudly if a training matrix's column order doesn't match the
    schema's feature vector for `model_name` ("flood" or "landslide").

    Call this right before fitting. A silent order mismatch corrupts
    predictions with no error thrown, per CLAUDE.md §3.1 — so this checks
    exact equality including position, not just membership.
    """
    expected = {"flood": flood_features, "landslide": landslide_features}.get(model_name)
    if expected is None:
        raise ValueError(f"Unknown model_name {model_name!r}; expected 'flood' or 'landslide'.")
    expected_order = expected()
    if list(columns) != expected_order:
        raise AssertionError(
            f"{model_name} feature order mismatch.\n"
            f"  schema.json expects: {expected_order}\n"
            f"  training matrix has: {list(columns)}\n"
            f"Fix the column order before fitting — do not proceed."
        )


def risk_tiers() -> dict[str, dict[str, Any]]:
    """The full risk_tiers dict: {tier_name: {ordinal, ml_threshold_pct, siren}}.
    Metadata keys (e.g. "_description") are stripped out."""
    return {k: v for k, v in load_schema()["risk_tiers"].items() if not k.startswith("_")}


def stale_after_seconds() -> int:
    """Seconds after which a node's last decision timestamp counts as stale."""
    return load_schema()["heartbeat"]["stale_after_seconds"]


def heartbeat_interval_seconds() -> int:
    """Expected seconds between heartbeats, for reference/display."""
    return load_schema()["heartbeat"]["interval_seconds"]


# NOTE: schema.json does not expose these weights as structured data — they
# only appear inside the free-text "_derivation" note on deep_soil_moisture_pct
# ("Weighted mean of the 30cm/60cm/100cm layers ... weights 0.25/0.45/0.30").
# There is no JSON field to read them from, so they are transcribed here by
# hand from that note. If schema.json's _derivation text ever changes, this
# constant must be updated to match — it will NOT update itself.
_DEEP_SOIL_WEIGHTS = {"30cm": 0.25, "60cm": 0.45, "100cm": 0.30}


def deep_soil_weights() -> dict[str, float]:
    """Weights for deriving deep_soil_moisture_pct from the 30/60/100cm layers.

    Transcribed from schema.json's free-text _derivation note (see comment
    above) — not read from a structured field, because none exists.
    """
    schema = load_schema()
    note = schema["sensor_reading"]["soil_moisture_profile"]  # touch it so a
    # missing profile section fails loudly rather than this function quietly
    # returning stale weights for a schema that no longer has them.
    del note
    return dict(_DEEP_SOIL_WEIGHTS)


def decision_object_keys() -> list[str]:
    """Keys the decision object (returned by /status, /history, /ingest) is
    documented to carry. Useful for defensively validating backend responses
    without hard-coding the key list a second time in the dashboard.
    Metadata keys (e.g. "_description") are stripped out."""
    return [k for k in load_schema()["decision_object"].keys() if not k.startswith("_")]


if __name__ == "__main__":
    # Quick manual sanity check: `python contract.py`
    print("schema path:", SCHEMA_PATH)
    print("flood features:", flood_features())
    print("landslide features:", landslide_features())
    print("risk tiers:", risk_tiers())
    print("stale after (s):", stale_after_seconds())
    print("deep soil weights:", deep_soil_weights())
    print("decision object keys:", decision_object_keys())
