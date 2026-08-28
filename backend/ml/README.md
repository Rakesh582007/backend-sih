# backend/ml — model training (scope: `feat/ml-tuning`)

Trains the two live regressors (flood, landslide) against the frozen contract
at `../../schema.json`. See `../CLAUDE.md` for full scope/rules. Summary:

- `contract.py` — the only place feature order, risk tiers, heartbeat, and the
  deep-soil weights are read from. `train.py` must import it, never retype
  these values.
- `train.py` — **not started yet.** Blocked on a real labelled dataset (see
  below). Do not train on synthetic data and present it as a working model —
  synthetic runs are a smoke test only, and must be labelled as such.
- `artifacts/` — trained `.pkl` files land here, and only here. Never written
  into `backend/models/` (that's the teammate's backend's territory; the
  handoff path is still TBD, see open items).

## Feature order (from schema.json `ml_feature_vectors`, order is load-bearing)

```
flood_model     = [rain_intensity_mm_hr, rain_duration_min, deep_soil_moisture_pct,
                    stream_depth_cm, rate_of_rise_cm_min]
landslide_model = [deep_soil_moisture_pct, slope_pitch_deg, pitch_rate_deg_min,
                    vibration_intensity, gyro_angular_vel_deg_s]
output          = float, percent probability, range [0, 100]
```

## Dataset

**Path: TBD.** No labelled dataset has been provided yet. `train.py` will
refuse to run against anything but a real, provided dataset — if it's not at
the path given, it stops and asks rather than inventing data.

## Metrics (once training runs)

TBD — will be filled in after a real run. Per the team's ML methodology:
plain accuracy is not reported (rare-event problem); expect regression/
threshold metrics plus F2 if framed as detection; spatially-blocked CV
(GroupKFold by district/grid) if spatial structure exists in the data, never
a naive random split; any resampling done inside the training fold only;
decision threshold tuned on the PR curve, not left at 0.5.

## Open data-integrity item — flag for judges, do not silently paper over

`rain_intensity_mm_hr` is specified in the schema as true mm/hour, but the
current hardware (a resistive rain plate) can only sense wetness/duration —
it has no way to measure rainfall *intensity*. **The true source of this
field is unresolved.** No conversion from wetness/duration to mm/hr has been
fabricated here, and none should be assumed downstream. This is open
question #3 in `../CLAUDE.md` §2, pending the teammate's answer.

## Other open items (see `../CLAUDE.md` §2 for full context)

- Whether `app.py` is retired or still runs — doesn't affect this directory
  either way; `ml/` only reads the schema and writes to `ml/artifacts/`.
- Where the teammate's backend will eventually load trained models from
  (path/filename) — TBD, so `artifacts/` output has no wiring to it yet.

## Honest caveat for judges (placeholder until a real run happens)

No model has been trained yet. Once one is, this line will be replaced with
the actual caveat, at minimum: "thresholds illustrative, pending real
catchment calibration."
