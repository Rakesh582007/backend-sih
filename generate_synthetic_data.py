"""
GiriKavach — Synthetic telemetry generator
===========================================

data/master_telemetry.csv did not exist (only an empty, misspelled
data/master_telementry.csv was present). This script produces the CSV that
train_tflm_models.py already expects, using the exact column names and
value ranges documented in that script's FEATURE_RANGES table.

Physically-motivated (not random-uniform) target formulas:
- flash_flood_probability rises with rain intensity/duration, stream depth,
  and rate of rise; soil already near saturation reduces infiltration and
  pushes more rain to runoff, so soil moisture is included too.
- landslide_probability rises with soil saturation (reduces shear strength),
  slope steepness, how fast the slope is deforming (pitch_rate), and
  vibration/gyro (ground movement / micro-shocks).

Each target = sigmoid(weighted sum of normalized features) * 100 + noise,
clipped to [0, 100]. This isn't a physical model, just a target function
with the right monotonic relationships so a tiny NN has real signal to learn.
"""

import numpy as np
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "data" / "master_telemetry.csv"

N_ROWS = 8000
SEED = 42

FEATURE_RANGES = {
    "rain_intensity_mm_hr": (0, 180),
    "rain_duration_min": (0, 720),
    "deep_soil_moisture_pct": (0, 100),
    "stream_depth_cm": (0, 500),
    "rate_of_rise_cm_min": (-5, 40),
    "slope_pitch_deg": (0, 65),
    "pitch_rate_deg_min": (0, 6),
    "vibration_intensity": (0, 100),
    "gyro_angular_vel_deg_s": (0, 30),
}


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def main():
    rng = np.random.default_rng(SEED)
    n = N_ROWS

    # Skewed-low distributions: most readings are calm, with a heavier tail
    # of storm/high-activity events, matching real sensor behavior.
    rain_intensity = rng.gamma(shape=1.5, scale=18, size=n).clip(0, 180)
    rain_duration = rng.gamma(shape=1.8, scale=60, size=n).clip(0, 720)

    # Soil moisture correlates with recent rain, plus its own baseline noise.
    soil_moisture = (
        20 + 0.35 * rain_intensity + 0.03 * rain_duration
        + rng.normal(0, 8, n)
    ).clip(0, 100)

    # Stream depth and rate-of-rise track rain intensity/duration/soil wetness.
    stream_depth = (
        10 + 1.4 * rain_intensity + 0.15 * rain_duration
        + 0.8 * soil_moisture + rng.normal(0, 25, n)
    ).clip(0, 500)
    rate_of_rise = (
        -1 + 0.09 * rain_intensity + 0.01 * rain_duration
        + rng.normal(0, 3, n)
    ).clip(-5, 40)

    # Slope/vibration sensors are largely independent of the rain channel,
    # with soil moisture giving a mild secondary influence (wet slopes creep).
    slope_pitch = rng.uniform(0, 65, n)
    pitch_rate = (
        0.02 * slope_pitch + 0.01 * soil_moisture + rng.gamma(1.2, 0.6, n)
    ).clip(0, 6)
    vibration = (rng.gamma(1.3, 12, n)).clip(0, 100)
    gyro_ang_vel = (0.15 * vibration + rng.gamma(1.2, 3, n)).clip(0, 30)

    df = pd.DataFrame({
        "rain_intensity_mm_hr": rain_intensity,
        "rain_duration_min": rain_duration,
        "deep_soil_moisture_pct": soil_moisture,
        "stream_depth_cm": stream_depth,
        "rate_of_rise_cm_min": rate_of_rise,
        "slope_pitch_deg": slope_pitch,
        "pitch_rate_deg_min": pitch_rate,
        "vibration_intensity": vibration,
        "gyro_angular_vel_deg_s": gyro_ang_vel,
    })

    def norm(col):
        lo, hi = FEATURE_RANGES[col]
        return ((df[col] - lo) / (hi - lo)).clip(0, 1)

    flood_z = (
        -6.5
        + 4.5 * norm("rain_intensity_mm_hr")
        + 2.5 * norm("rain_duration_min")
        + 1.5 * norm("deep_soil_moisture_pct")
        + 3.0 * norm("stream_depth_cm")
        + 3.5 * norm("rate_of_rise_cm_min")
    )
    flood_prob = (sigmoid(flood_z) * 100 + rng.normal(0, 4, n)).clip(0, 100)

    landslide_z = (
        -6.0
        + 2.5 * norm("deep_soil_moisture_pct")
        + 3.0 * norm("slope_pitch_deg")
        + 3.5 * norm("pitch_rate_deg_min")
        + 2.0 * norm("vibration_intensity")
        + 2.0 * norm("gyro_angular_vel_deg_s")
    )
    landslide_prob = (sigmoid(landslide_z) * 100 + rng.normal(0, 4, n)).clip(0, 100)

    df["flash_flood_probability"] = flood_prob
    df["landslide_probability"] = landslide_prob

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Wrote {len(df)} rows -> {DATA_PATH}")
    print(df.describe().T[["min", "mean", "max"]])


if __name__ == "__main__":
    main()
