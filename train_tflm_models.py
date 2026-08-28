"""
GiriKavach — TFLM (offline path) Model Training + Export
===========================================================

This is the OFFLINE inference path: a tiny neural net small enough to run
directly on the Node 2 ESP32 via TensorFlow Lite Micro, with zero dependency
on the laptop/FastAPI/XGBoost pipeline being alive.

Design notes (why this is deliberately NOT a copy of the XGBoost pipeline):
- XGBoost (train_models.py) cannot be converted to TFLM — tree ensembles
  have no TFLM op kernels. So this trains a *separate*, much smaller model
  family (2-layer MLP) from the same synthetic telemetry.
- Same 5 features per hazard as the XGBoost models, so Node 2 can run
  either engine off the identical LoRa+ultrasonic packet it already has —
  "offline" here means "no laptop", not "less data".
- int8 quantization keeps each model under ~5-10 KB, comfortably inside
  ESP32 SRAM alongside the LoRa/GSM/sensor driver code.
- Two independent single-output models (not one shared multi-output model)
  on purpose: if the LoRa link from Node 1 drops, Node 2 still has its own
  ultrasonic reading, but note both models below still need Node 1's
  features too (rain/soil/slope/vibration/gyro all originate at Node 1) —
  see the firmware skeleton's comment on link-loss behavior.

Output: models/tflm/flood_model.cc, models/tflm/landslide_model.cc
        (+ matching .h files) ready to #include in ESP32 firmware.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from pathlib import Path

# __file__ is "the path to this script itself". This script lives at the
# project root (no data_pipeline/ subfolder), so data/ and models/ are
# direct children of its own directory — not one level up.
SCRIPT_DIR = Path(__file__).resolve().parent      # project root
PROJECT_ROOT = SCRIPT_DIR
DATA_PATH = PROJECT_ROOT / "data" / "master_telemetry.csv"
OUT_DIR = PROJECT_ROOT / "models" / "tflm"

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

# Normalization ranges (feature min/max) — MUST be baked into firmware too,
# since the ESP32 has to apply the same scaling to raw sensor reads before
# feeding the quantized model. Printed at the end of this script.
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


def normalize(df, features):
    out = df[features].copy()
    for f in features:
        lo, hi = FEATURE_RANGES[f]
        out[f] = ((out[f] - lo) / (hi - lo)).clip(0, 1)
    return out.values.astype("float32")


def build_tiny_model(n_features: int) -> tf.keras.Model:
    # Deliberately tiny: this has to fit + run fast on an ESP32 (no FPU
    # acceleration to speak of). 2 hidden layers, <500 params total.
    inputs = tf.keras.Input(shape=(n_features,))
    x = tf.keras.layers.Dense(12, activation="relu")(inputs)
    x = tf.keras.layers.Dense(8, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)  # 0-1, scale to 0-100 in firmware
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def quantize_and_export(model: tf.keras.Model, X_train: np.ndarray, name: str):
    def representative_dataset():
        for i in range(min(200, len(X_train))):
            yield [X_train[i : i + 1]]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # TF's default int8 quantization makes Dense/FULLY_CONNECTED weights
    # per-channel (one scale per output unit). Found on real hardware: the
    # Chirale_TensorFlowLite Arduino library (a July 2024 TFLM snapshot)
    # silently mis-executes per-channel FULLY_CONNECTED — no error, just a
    # wrong answer that only showed up by comparing device output against
    # Python's interpreter for byte-identical int8 input. Forcing per-tensor
    # quantization for Dense layers trades a little accuracy for correctness
    # on this specific on-device runtime.
    converter._experimental_disable_per_channel_quantization_for_dense_layers = True
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    tflite_path = f"{OUT_DIR}/{name}.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    # Emit a C header with the model as a byte array (equivalent to `xxd -i`)
    var_name = f"g_{name}_data"
    c_lines = [
        f'#include "{name}.h"',
        "",
        f"alignas(8) const unsigned char {var_name}[] = {{",
    ]
    hex_bytes = [f"0x{b:02x}" for b in tflite_model]
    for i in range(0, len(hex_bytes), 12):
        c_lines.append("    " + ", ".join(hex_bytes[i : i + 12]) + ",")
    c_lines.append("};")
    c_lines.append(f"const int {var_name}_len = {len(tflite_model)};")

    with open(f"{OUT_DIR}/{name}.cc", "w") as f:
        f.write("\n".join(c_lines) + "\n")

    h_lines = [
        f"#ifndef {name.upper()}_MODEL_H_",
        f"#define {name.upper()}_MODEL_H_",
        "",
        f"extern const unsigned char {var_name}[];",
        f"extern const int {var_name}_len;",
        "",
        "#endif",
    ]
    with open(f"{OUT_DIR}/{name}.h", "w") as f:
        f.write("\n".join(h_lines) + "\n")

    size_kb = len(tflite_model) / 1024
    print(f"[{name}] quantized model: {size_kb:.1f} KB -> {OUT_DIR}/{name}.cc / .h")
    return tflite_path


def train_and_export(df, features, target, name):
    X = normalize(df, features)
    y = (df[target] / 100.0).values.astype("float32")  # scale target to 0-1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_tiny_model(len(features))
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=40, batch_size=32, verbose=0)

    val_mae = model.evaluate(X_test, y_test, verbose=0)[1]
    print(f"[{name}] float model val MAE: {val_mae*100:.2f} points (0-100 scale)")

    tflite_path = quantize_and_export(model, X_train, name)

    # Sanity-check the quantized model actually still predicts sensibly
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]
    scale, zero_point = in_detail["quantization"]

    preds = []
    for i in range(len(X_test)):
        x_q = (X_test[i : i + 1] / scale + zero_point).astype(in_detail["dtype"])
        interpreter.set_tensor(in_detail["index"], x_q)
        interpreter.invoke()
        out_q = interpreter.get_tensor(out_detail["index"])
        out_scale, out_zero = out_detail["quantization"]
        preds.append((out_q.astype("float32") - out_zero) * out_scale)
    preds = np.array(preds).flatten()
    quant_mae = np.mean(np.abs(preds - y_test)) * 100
    print(f"[{name}] int8-quantized model val MAE: {quant_mae:.2f} points (0-100 scale)\n")


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    train_and_export(df, FLOOD_FEATURES, FLOOD_TARGET, "flood_model")
    train_and_export(df, LANDSLIDE_FEATURES, LANDSLIDE_TARGET, "landslide_model")

    print("Feature normalization ranges (must match firmware-side scaling exactly):")
    for k, v in FEATURE_RANGES.items():
        print(f"  {k}: {v}")