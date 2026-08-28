"""
GiriKavach — XGBoost live-inference model training
=====================================================

Trains the two regressors main.py loads at startup (models/flood_model.pkl,
models/landslide_model.pkl). These are DIFFERENT from the TFLM models
(train_tflm_models.py) — same feature/target schema and same synthetic
data, but a full-size XGBoost regressor for the laptop/FastAPI "online"
path, not a quantized tiny net for the ESP32 offline path. See main.py's
module docstring for why both exist.

Trained on the same data/master_telemetry.csv the TFLM path uses — see
generate_synthetic_data.py for how that data was constructed and its
caveats (physically-motivated synthetic targets, not real historical
events). These are demo-quality models: real deployment would need
actual historical flood/landslide records, not synthetic data.
"""

from __future__ import annotations

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "data" / "master_telemetry.csv"
MODEL_DIR = SCRIPT_DIR / "models"

FLOOD_FEATURES = [
    "rain_intensity_mm_hr",
    "rain_duration_min",
    "deep_soil_moisture_pct",
    "stream_depth_cm",
    "rate_of_rise_cm_min",
]
FLOOD_TARGET = "flash_flood_probability"

LANDSLIDE_FEATURES = [
    "deep_soil_moisture_pct",
    "slope_pitch_deg",
    "pitch_rate_deg_min",
    "vibration_intensity",
    "gyro_angular_vel_deg_s",
]
LANDSLIDE_TARGET = "landslide_probability"


def train_and_save(df: pd.DataFrame, features: list[str], target: str, name: str) -> None:
    X, y = df[features], df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        objective="reg:squarederror",
    )
    model.fit(X_train, y_train)

    mae = mean_absolute_error(y_test, model.predict(X_test))
    print(f"[{name}] val MAE: {mae:.2f} points (0-100 scale)")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODEL_DIR / f"{name}.pkl"
    joblib.dump(model, out_path)
    print(f"[{name}] saved -> {out_path}")


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    train_and_save(df, FLOOD_FEATURES, FLOOD_TARGET, "flood_model")
    train_and_save(df, LANDSLIDE_FEATURES, LANDSLIDE_TARGET, "landslide_model")
