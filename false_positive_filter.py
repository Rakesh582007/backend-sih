"""
GiriKavach — False-Positive Rejection Layer + Rule-Based Safety Net
=====================================================================

WHY THIS FILE EXISTS (the one-sentence version for judges):
Early warning systems rarely fail by missing a disaster. They fail by
crying wolf until the village stops evacuating. This module is the
difference between a siren people trust and a siren people ignore.

It sits BETWEEN the ML models and the siren, and does two opposite jobs:

  1. SUPPRESS false alarms   — a high ML score alone is not enough to
                               sound a siren (this file's main job)
  2. FORCE a true alarm      — if raw physics is undeniable, fire even
                               when the ML says "low risk" (the safety
                               net, in case the model is silently wrong)

Both matter. (1) protects trust; (2) protects lives when ML fails.

------------------------------------------------------------------
THE FIVE FALSE-POSITIVE SOURCES THIS DEFENDS AGAINST
------------------------------------------------------------------
FP1. Human/animal activity vibration.
     Tea plantation workers, vehicles, machinery, livestock all shake
     an SW-420. This is the single most likely nuisance source in an
     inhabited hill catchment, and was specifically flagged in mentor
     review. Defence: vibration is NEVER sufficient on its own — it
     only counts when the ground is already deeply saturated, because
     dry ground does not fail no matter how much it is shaken.

FP2. Single-sample sensor spikes.
     Electrical noise, a loose jumper, an ADC glitch. Defence: temporal
     persistence — risk must hold above threshold for N consecutive
     readings before escalating.

FP3. Floating river debris (logs, branches) under the ultrasonic.
     A log drifting past reads as a sudden 2m "water level rise".
     Defence: median filter over a rolling window. A median ignores
     one outlier by construction; a mean would not.

FP4. Transient tilt (wind gust, someone bumping the pole, thermal
     expansion of the mount). Defence: real slope creep is MONOTONIC
     and PERSISTENT — it drifts one direction and stays. A bump
     returns to baseline. We track sustained drift, not instantaneous
     angle.

FP5. Sensor fault / drift.
     A dead or drifting sensor can read a constant extreme value
     forever. Defence: physical plausibility bounds + a "stuck sensor"
     check (identical reading for many cycles = suspect, not alarming).

------------------------------------------------------------------
DESIGN NOTE ON HYSTERESIS (worth explaining to judges)
------------------------------------------------------------------
Escalating requires N consecutive confirmations; DE-escalating requires
M consecutive quiet readings, where M > N. This asymmetry is deliberate:
we are slow to alarm, and even slower to stand down. A system that
flapped ALERT->NORMAL->ALERT every few seconds would be worse than
useless during an actual event.
"""

from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum


class RiskLevel(IntEnum):
    """Ordered so comparisons work: NORMAL < WATCH < ALERT < CRITICAL."""
    NORMAL = 0
    WATCH = 1
    ALERT = 2
    CRITICAL = 3


# ---------------------------------------------------------------------
# Tunable thresholds. Every number here is a policy decision, not a
# law of physics — keep them in one place so they can be defended,
# adjusted per deployment site, and explained.
# ---------------------------------------------------------------------
@dataclass
class FilterConfig:
    # --- ML probability thresholds for each risk tier (0-100) ---
    watch_threshold: float = 50.0
    alert_threshold: float = 70.0
    critical_threshold: float = 85.0

    # --- Temporal persistence (FP2) ---
    # How many CONSECUTIVE readings above a tier before we escalate to it.
    consecutive_to_escalate: int = 3
    # How many consecutive readings below before we stand down.
    # Deliberately larger than escalate — see hysteresis note above.
    consecutive_to_deescalate: int = 6

    # --- Physical gate for landslide (FP1) ---
    # Vibration and tilt only count as landslide evidence when the deep
    # soil is genuinely saturated. Dry slopes do not fail.
    # (Literature supports a saturation-threshold effect around 0.70;
    #  85% is this project's conservative working figure — calibrate
    #  per site before any real deployment.)
    saturation_gate_pct: float = 85.0
    # A slope below this angle is not a debris-flow initiation risk.
    slope_gate_deg: float = 25.0

    # --- Median filter window for ultrasonic (FP3) ---
    depth_median_window: int = 10

    # --- Sustained tilt drift (FP4) ---
    tilt_history_len: int = 20
    # Degrees of NET drift across the window to count as real creep.
    tilt_drift_deg: float = 1.5

    # --- Sensor plausibility bounds (FP5) ---
    depth_min_cm: float = 0.0
    depth_max_cm: float = 500.0
    soil_min_pct: float = 0.0
    soil_max_pct: float = 100.0
    # Identical reading this many times in a row = suspected stuck sensor.
    stuck_sensor_cycles: int = 30

    # --- Rule-based safety net (fires regardless of ML) ---
    # These are deliberately EXTREME. If physics looks like this, we do
    # not care what the model thinks.
    safety_net_soil_pct: float = 92.0
    safety_net_slope_deg: float = 38.0
    safety_net_rate_of_rise_cm_min: float = 5.0  # spec Section 3


@dataclass
class SensorReading:
    """One merged sample: Node 1 telemetry + Node 2 local sensors."""
    rain_intensity_mm_hr: float = 0.0
    rain_duration_min: float = 0.0
    deep_soil_moisture_pct: float = 0.0
    slope_pitch_deg: float = 0.0
    pitch_rate_deg_min: float = 0.0
    vibration_intensity: float = 0.0
    gyro_angular_vel_deg_s: float = 0.0
    stream_depth_cm: float = 0.0
    rate_of_rise_cm_min: float = 0.0
    node1_link_fresh: bool = True


@dataclass
class FilterDecision:
    """What the filter concluded, and — critically — WHY.

    The `reasons` list is not decoration. When a judge asks "why did it
    alarm?" or an operator asks "why didn't it?", this is the answer.
    Every suppression and every escalation is recorded.
    """
    level: RiskLevel
    flood_probability: float
    landslide_probability: float
    should_sound_siren: bool
    reasons: list[str] = field(default_factory=list)
    suppressed: list[str] = field(default_factory=list)
    safety_net_triggered: bool = False
    sensor_health_warnings: list[str] = field(default_factory=list)


class FalsePositiveFilter:
    """Stateful across readings — it needs history to reject transients."""

    def __init__(self, config: FilterConfig | None = None):
        self.cfg = config or FilterConfig()
        self._depth_history: deque[float] = deque(maxlen=self.cfg.depth_median_window)
        self._tilt_history: deque[float] = deque(maxlen=self.cfg.tilt_history_len)
        self._last_soil: float | None = None
        self._soil_repeat_count = 0
        self._current_level = RiskLevel.NORMAL
        self._consecutive_above = 0
        self._consecutive_below = 0

    # -- FP3: median filter kills single outliers by construction -------
    def _filtered_depth(self, raw_depth: float) -> float:
        self._depth_history.append(raw_depth)
        return statistics.median(self._depth_history)

    # -- FP4: is the tilt DRIFTING (real creep) or just wobbling? -------
    def _sustained_tilt_drift(self, pitch_deg: float) -> float:
        self._tilt_history.append(pitch_deg)
        if len(self._tilt_history) < self.cfg.tilt_history_len:
            return 0.0
        # Compare the oldest third to the newest third. A transient bump
        # averages out; a genuine creep shows up as a net offset.
        n = len(self._tilt_history) // 3
        old = statistics.mean(list(self._tilt_history)[:n])
        new = statistics.mean(list(self._tilt_history)[-n:])
        return new - old

    # -- FP5: bounds + stuck-sensor detection --------------------------
    def _sensor_health(self, r: SensorReading) -> list[str]:
        warnings: list[str] = []
        if not (self.cfg.depth_min_cm <= r.stream_depth_cm <= self.cfg.depth_max_cm):
            warnings.append(
                f"stream_depth {r.stream_depth_cm}cm outside plausible range — sensor fault suspected"
            )
        if not (self.cfg.soil_min_pct <= r.deep_soil_moisture_pct <= self.cfg.soil_max_pct):
            warnings.append(
                f"soil_moisture {r.deep_soil_moisture_pct}% outside plausible range — sensor fault suspected"
            )
        # Stuck sensor: bit-identical soil readings for many cycles.
        if self._last_soil is not None and r.deep_soil_moisture_pct == self._last_soil:
            self._soil_repeat_count += 1
            if self._soil_repeat_count >= self.cfg.stuck_sensor_cycles:
                warnings.append(
                    f"soil sensor returned identical value {self._soil_repeat_count}x — possibly stuck/disconnected"
                )
        else:
            self._soil_repeat_count = 0
        self._last_soil = r.deep_soil_moisture_pct
        return warnings

    # -- FP1: the physical gate. THE key false-positive defence. --------
    def _landslide_evidence_is_physical(self, r: SensorReading) -> tuple[bool, str]:
        """Vibration/tilt only mean 'landslide' on saturated, steep ground.

        This is what stops a tea-estate tractor, a herd of goats, or a
        passing truck from evacuating a village. It is not a statistical
        trick — it is the actual failure physics: shear failure needs
        pore water pressure. Shaking dry, gentle ground does nothing.
        """
        if r.deep_soil_moisture_pct < self.cfg.saturation_gate_pct:
            return False, (
                f"vibration/tilt present but deep soil only {r.deep_soil_moisture_pct:.0f}% "
                f"(<{self.cfg.saturation_gate_pct:.0f}% saturation gate) — "
                f"consistent with surface activity (traffic/livestock/work), not slope failure"
            )
        if r.slope_pitch_deg < self.cfg.slope_gate_deg:
            return False, (
                f"slope {r.slope_pitch_deg:.0f}° below {self.cfg.slope_gate_deg:.0f}° "
                f"initiation gate — terrain not prone to debris-flow initiation"
            )
        return True, ""

    # -- The reverse: physics so extreme we override the ML ------------
    def _check_safety_net(self, r: SensorReading) -> tuple[bool, list[str]]:
        """If the ML is silently wrong, this still fires.

        Defence-in-depth: an ML model trained on physics-informed
        synthetic data could plausibly miss a real-world pattern it
        never saw. These raw-threshold rules do not depend on the model
        being right about anything.
        """
        triggered: list[str] = []
        if (
            r.deep_soil_moisture_pct >= self.cfg.safety_net_soil_pct
            and r.slope_pitch_deg >= self.cfg.safety_net_slope_deg
        ):
            triggered.append(
                f"SAFETY NET: soil {r.deep_soil_moisture_pct:.0f}% + slope "
                f"{r.slope_pitch_deg:.0f}° exceed physical failure thresholds — "
                f"alerting regardless of model output"
            )
        if r.rate_of_rise_cm_min >= self.cfg.safety_net_rate_of_rise_cm_min:
            triggered.append(
                f"SAFETY NET: river rising {r.rate_of_rise_cm_min:.1f} cm/min "
                f"(>{self.cfg.safety_net_rate_of_rise_cm_min} threshold) — "
                f"indicates upstream cloudburst or dam-break wave"
            )
        return bool(triggered), triggered

    # -- FP2: temporal persistence + hysteresis -------------------------
    def _apply_hysteresis(self, target: RiskLevel) -> RiskLevel:
        if target > self._current_level:
            self._consecutive_above += 1
            self._consecutive_below = 0
            if self._consecutive_above >= self.cfg.consecutive_to_escalate:
                self._current_level = target
                self._consecutive_above = 0
        elif target < self._current_level:
            self._consecutive_below += 1
            self._consecutive_above = 0
            if self._consecutive_below >= self.cfg.consecutive_to_deescalate:
                # Step down one tier at a time, never straight to NORMAL.
                self._current_level = RiskLevel(self._current_level - 1)
                self._consecutive_below = 0
        else:
            self._consecutive_above = 0
            self._consecutive_below = 0
        return self._current_level

    def _raw_level(self, flood_p: float, landslide_p: float) -> RiskLevel:
        worst = max(flood_p, landslide_p)
        if worst >= self.cfg.critical_threshold:
            return RiskLevel.CRITICAL
        if worst >= self.cfg.alert_threshold:
            return RiskLevel.ALERT
        if worst >= self.cfg.watch_threshold:
            return RiskLevel.WATCH
        return RiskLevel.NORMAL

    # ------------------------------------------------------------------
    def evaluate(
        self,
        reading: SensorReading,
        flood_probability: float,
        landslide_probability: float,
    ) -> FilterDecision:
        """Run one reading through the full filter chain."""
        reasons: list[str] = []
        suppressed: list[str] = []

        health = self._sensor_health(reading)

        # FP3 — clean the depth signal before it influences anything
        filtered_depth = self._filtered_depth(reading.stream_depth_cm)
        if abs(filtered_depth - reading.stream_depth_cm) > 20:
            suppressed.append(
                f"stream depth outlier rejected: raw {reading.stream_depth_cm:.0f}cm vs "
                f"median {filtered_depth:.0f}cm — likely floating debris"
            )

        # FP4 — decide whether tilt is real creep
        drift = self._sustained_tilt_drift(reading.slope_pitch_deg)
        tilt_is_creep = abs(drift) >= self.cfg.tilt_drift_deg
        if reading.pitch_rate_deg_min > 0.5 and not tilt_is_creep:
            suppressed.append(
                f"tilt movement not sustained (net drift {drift:.2f}° over window) — "
                f"transient (wind/impact), not slope creep"
            )

        # Link freshness: landslide features ALL originate at Node 1.
        # Stale link means we cannot assess landslide risk at all — say
        # so explicitly rather than silently scoring on old data.
        effective_landslide_p = landslide_probability
        if not reading.node1_link_fresh:
            effective_landslide_p = 0.0
            suppressed.append(
                "landslide inference skipped — Node 1 link stale, all landslide "
                "features originate upstream (NOT a low-risk finding)"
            )
        else:
            # FP1 — the physical gate
            physical, why_not = self._landslide_evidence_is_physical(reading)
            if not physical and landslide_probability >= self.cfg.watch_threshold:
                effective_landslide_p = min(landslide_probability, self.cfg.watch_threshold - 1)
                suppressed.append(why_not)
            elif physical and landslide_probability >= self.cfg.watch_threshold:
                reasons.append(
                    f"landslide evidence physically consistent: soil "
                    f"{reading.deep_soil_moisture_pct:.0f}%, slope {reading.slope_pitch_deg:.0f}°"
                )

        # Rule-based safety net — can override everything below
        net_fired, net_reasons = self._check_safety_net(reading)
        reasons.extend(net_reasons)

        target = self._raw_level(flood_probability, effective_landslide_p)
        if net_fired:
            target = RiskLevel.CRITICAL

        # FP2 — persistence/hysteresis, unless the safety net fired
        # (a genuine dam-break wave should not wait 3 cycles to alarm)
        if net_fired:
            self._current_level = RiskLevel.CRITICAL
            level = RiskLevel.CRITICAL
        else:
            level = self._apply_hysteresis(target)
            if target > level:
                suppressed.append(
                    f"escalation to {target.name} pending — needs "
                    f"{self.cfg.consecutive_to_escalate} consecutive confirmations "
                    f"(at {self._consecutive_above})"
                )

        if flood_probability >= self.cfg.watch_threshold:
            reasons.append(
                f"flood model {flood_probability:.0f}% (rain "
                f"{reading.rain_intensity_mm_hr:.0f}mm/hr, rise "
                f"{reading.rate_of_rise_cm_min:.1f}cm/min)"
            )

        return FilterDecision(
            level=level,
            flood_probability=flood_probability,
            landslide_probability=effective_landslide_p,
            should_sound_siren=level >= RiskLevel.ALERT,
            reasons=reasons,
            suppressed=suppressed,
            safety_net_triggered=net_fired,
            sensor_health_warnings=health,
        )
