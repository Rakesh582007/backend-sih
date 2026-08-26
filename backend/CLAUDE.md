# SIH 26192 — Ridge & Valley Flood/Landslide Backend

## What this system actually does

Two hardware nodes talk to each other over LoRa (no internet needed on the ridge):

- **Node A (ridge)** — 5 soil moisture sensors at different depths, a tilt sensor (MPU6050), a rain plate. Sends a small packet over LoRa every 30s. No WiFi.
- **Node B (village gateway)** — receives Node A's LoRa packets, also runs its own ultrasonic water-level sensor + rain plate. Node B has WiFi + a SIM800L GSM module + a buzzer. **This backend is what Node B talks to.**

Node B POSTs a combined JSON payload to our backend every cycle. Our job:
1. Store it
2. Clean it (reject bad readings, fill gaps if a sensor dies)
3. Decide the risk level (NORMAL / WATCH / ALERT / CRITICAL)
4. Push that to a dashboard + trigger alerts

**Important:** Node B also runs this exact same decision logic *locally on the ESP32* as a fallback — if it can't reach our backend for 15+ seconds, it fires the buzzer/SMS itself without us. Our backend is the "smart, connected" path; the ESP32 firmware is the "dumb, always-works" path. We are not the single point of failure, which is the whole point.

---

## The ML/AI — explained simply, no jargon

This confused you before because I originally described it wrong. Here's the real, correct picture. **There are two completely different kinds of "smart" in this system, and they answer two different questions.**

### 1. The live decision (WATCH/ALERT/CRITICAL) is NOT machine learning

It's a **rule table** (a "trigger matrix") using real numbers from your manual. Example — landslide:

| Level | Rain duration | Deep soil moisture | Tilt | 
|---|---|---|---|
| WATCH | >30 min | >70% | — |
| ALERT | >60 min | >85% | Δ>0.5° |
| CRITICAL | any | >85% | Δ>2.0° or fast-rising |

**All conditions in a row must be true together.** This is deterministic if-statements, not AI. Why not ML here? Because a judge can ask "why did it alert?" and you can give an exact, defensible answer instead of "the model said so." This is a *feature*, not a shortcut — explainability is a real design principle for safety systems.

Same idea for flood: convert ultrasonic distance → water height → rate of rise (dh/dt), compare against fixed thresholds. Also not ML.

### 2. The ML model answers a completely different question: "is this hillside dangerous *in general*?"

This is the **susceptibility model** — trained once (before the hackathon, using your laptop) on historical landslide records + terrain data (slope, elevation, etc.). It outputs one static number per location, 0 to 1, that never changes in real time.

- **Algorithm: LightGBM or XGBoost** (gradient boosting) — not deep learning. This is deliberate: for this kind of tabular terrain data, boosted trees beat neural nets, train in seconds, and give you feature importance you can show on a slide.
- Trained on NASA COOLR / GSI Bhukosh (labels: where landslides happened) + SRTM DEM (terrain: slope, elevation).
- Output: a single "danger score" per node location, computed once, loaded as a `.pkl` file at backend startup.

**The real combination:** `final_risk = susceptibility_score × live_trigger_level`. A saturated sensor on a historically stable slope is treated differently than the same reading on a slope that's failed before. That's your "multi-source data fusion" story from the problem statement — done honestly.

### 3. TensorFlow — where it actually fits (since you want to use it)

The one place a neural net is genuinely appropriate: **short-term forecasting** — "will the water keep rising for the next few hours?" This is a time-series problem (LSTM), separate from the susceptibility model. It's a stretch goal, not core logic — build it after Phases 1-4 below are working.

### One-sentence summary you can say out loud
*"Machine learning tells us where a slope is dangerous in general, using historical data. A rule engine tells us if something dangerous is happening right now, using live sensors. We multiply the two together. TensorFlow adds a forecast layer on top — 'is it going to get worse.'"*

---

## Tech stack (corrected)

| Piece | Tool | Why |
|---|---|---|
| API framework | **FastAPI** (not Flask) | Manual specifies it; also gives free request validation |
| Server | Uvicorn | Standard FastAPI runner |
| Database | SQLite + SQLAlchemy | Zero-setup, file-based, fine for a hackathon |
| Susceptibility model | LightGBM or XGBoost | Best for tabular terrain data, trains fast |
| Forecasting (stretch) | TensorFlow/Keras (LSTM) | Your requirement, used for the forecast layer only |
| Dashboard | Streamlit | Manual specifies it (not React) |
| Alerts | Web Push + optional Ollama for alert text | Local, no internet needed |

---

## Folder structure

```
backend/
├── app.py                 # FastAPI app, defines /ingest and /status/{village_id}
├── db.py                  # SQLAlchemy models: SensorReading, NodeStatus
├── pipeline.py             # Self-healing: Z-score outlier rejection, dead-node marking, IDW fill
├── risk_engine.py          # Trigger matrices (flood + landslide) — the rule logic from section 04
├── susceptibility.py       # Loads the trained LightGBM .pkl, combines with live risk
├── forecast.py             # (stretch) TensorFlow LSTM for rainfall/water-level forecasting
├── alerts.py               # Web push + optional Ollama alert text generation
├── models/
│   └── susceptibility_model.pkl
├── dashboard/
│   └── streamlit_app.py
└── requirements.txt
```

---

## Step-by-step workflow (do these in order)

### Phase 1 — FastAPI skeleton (~1-2h)
1. `pip install fastapi uvicorn sqlalchemy`
2. Build `/ingest` (POST) — accepts Node B's JSON payload
3. Build `/status/{village_id}` (GET) — returns latest state, for dashboard polling
4. Test with fake JSON via `curl` or Postman — **you don't need real hardware yet to build this**

### Phase 2 — Database (~1h)
1. Define `SensorReading` table: node_id, timestamp, 5x soil values, tilt, rain_state, water_level, battery
2. Define `NodeStatus` table: node_id, last_seen, is_dead (bool)
3. Wire `/ingest` to write into the DB

### Phase 3 — Self-healing pipeline (~2h)
1. Z-score check per channel — reject a reading that's a wild statistical outlier
2. If a node hasn't sent data in >15s (per manual's own offline threshold), mark it `DEAD`
3. If a soil sensor in the depth array is dead, estimate its value using inverse-distance weighting (IDW) from its depth neighbours — this is the "self-healing" the manual specifically wants

### Phase 4 — Risk engine (~2-3h) — the most important part
1. Implement the **flood** dh/dt math exactly as in the manual: median-filter the last 5 pings, fit a regression slope over 60s, compare to WATCH/ALERT/CRITICAL thresholds
2. Implement the **landslide** trigger matrix: weighted deep soil moisture (S4×0.4 + S5×0.6), tilt delta from install baseline, rain duration — all three conditions must hold together per level
3. Return one of NORMAL / WATCH / ALERT / CRITICAL from a single function

### Phase 5 — Susceptibility model (~2-4h, can run in parallel with Phase 4)
1. Download NASA COOLR + SRTM DEM data
2. Train LightGBM with **spatial cross-validation** (GroupKFold by district — not random split, or your score is fake)
3. Optimize for **F2**, not accuracy (rare-event problem — accuracy is meaningless here)
4. Save as `.pkl`, load it in the backend, multiply with live risk level

### Phase 6 — Dashboard (~2-3h)
1. Streamlit app polling `/status/{village_id}`
2. The depth profile as a vertical strip (this is your "money shot" per the manual — shows the wetting front visually)
3. Tilt trace, water level + rate, per-sensor health strip

### Phase 7 — Alerts (~1-2h)
1. Web Push wired to the CRITICAL branch
2. (Optional) Ollama generating human-readable alert text from the risk state

### Phase 8 — Stretch: TensorFlow forecasting (only if time remains)
1. LSTM trained on rainfall/water-level time series
2. Predicts "will this keep rising for the next N hours"

---

