"""Degradation forecasting — the centerpiece of the twin (BUILD_SPEC.md §6).

Projects WHEN a drifting signal will cross its spec limit, with an honest
confidence interval, and explicitly refuses to forecast through step changes.
No ML: a changepoint-aware, EWMA-smoothed, Theil-Sen robust linear fit.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from scipy import stats

EWMA_ALPHA = 0.2
WINDOW_MAX_SAMPLES = 300
# BUILD_SPEC.md §6.2 step 1 literally says "4 hours" — that value makes forecasting
# structurally impossible for tier-B signals: at their 1-in-5-cycle sampling rate
# (~5-6 min/sample), a 4h window can only ever hold ~35-45 samples, permanently
# below MIN_SAMPLES=60 no matter how long a real drift continues (older readings
# age out of the window as fast as new ones arrive). Raised to 8h, which comfortably
# covers tier B's cadence (needs ~6h to reach 60 samples) while tier A/C are
# unaffected since WINDOW_MAX_SAMPLES=300 binds first for them well under 8h.
WINDOW_MAX_HOURS = 8.0
MIN_SAMPLES = 60
MAX_HORIZON_HOURS = 72.0
CHANGEPOINT_RECENT_SAMPLES = 30
PERSISTENCE_TICKS = 10            # consecutive significant evaluations required before surfacing
RESIDUAL_SPREAD_FACTOR = 3.0     # residual std vs expected EWMA-smoothed noise sd
CURVATURE_FACTOR = 1.3           # second-half slope must exceed this multiple of first-half
CURVATURE_MIN_HALF_SAMPLES = 15

STATUS_STABLE = "stable"
STATUS_DRIFTING = "drifting"
STATUS_FORECASTABLE = "forecastable"
STATUS_CROSSED = "crossed"
STATUS_UNSTABLE = "unstable"

TIER_CONFIDENCE = {"A": 1.0, "B": 0.6, "C": 0.3}


@dataclass
class DegradationForecast:
    station_id: str
    signal: str
    current_value: float
    slope_per_hour: float
    slope_ci: tuple
    spec_limit: Optional[float]
    hours_to_limit: Optional[float]
    hours_to_limit_ci: Optional[tuple]
    confidence: float
    status: str
    reason: str
    n_samples: int
    window_start_ts: Optional[datetime]


# ---------------------------------------------------------------------------
# Ring buffer
# ---------------------------------------------------------------------------

class SignalBuffer:
    """Rolling window of (ts, value) readings for one (station, signal) pair."""

    def __init__(self, max_samples: int = WINDOW_MAX_SAMPLES, max_hours: float = WINDOW_MAX_HOURS):
        self.readings: list[tuple[datetime, float]] = []
        self.max_samples = max_samples
        self.max_hours = max_hours

    def add(self, ts: datetime, value: float) -> None:
        self.readings.append((ts, value))
        self._trim()

    def _trim(self) -> None:
        if not self.readings:
            return
        if len(self.readings) > self.max_samples:
            self.readings = self.readings[-self.max_samples:]
        cutoff = self.readings[-1][0] - timedelta(hours=self.max_hours)
        self.readings = [r for r in self.readings if r[0] >= cutoff]


# ---------------------------------------------------------------------------
# Changepoint detection (CUSUM-style)
# ---------------------------------------------------------------------------

def detect_changepoint(values: np.ndarray) -> Optional[int]:
    """Returns the index of a detected level shift, or None. Uses the CUSUM
    argmax method: the index that maximizes cumulative deviation from the
    window mean is the best single-changepoint candidate; a shift is only
    reported if the before/after means differ materially relative to noise."""
    n = len(values)
    if n < 10:
        return None
    mean = values.mean()
    cusum = np.cumsum(values - mean)
    idx = int(np.argmax(np.abs(cusum)))
    if idx < 3 or idx > n - 4:
        return None
    before, after = values[:idx], values[idx:]
    # Within-group std, NOT std of the concatenated array — the latter is inflated
    # by the very shift being tested for (it's bimodal), which silently raises the
    # detection threshold in proportion to the shift and can make large, obvious
    # step changes undetectable. This was a real bug, not a tuning choice.
    pooled_std = float(np.sqrt((np.var(before) + np.var(after)) / 2)) + 1e-9
    shift = abs(before.mean() - after.mean())
    if shift > 1.3 * pooled_std and len(after) >= 5:
        return idx
    return None


def _ewma(values: np.ndarray, alpha: float = EWMA_ALPHA) -> np.ndarray:
    out = np.empty_like(values, dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def _elapsed_hours(timestamps: list[datetime]) -> np.ndarray:
    t0 = timestamps[0]
    return np.array([(t - t0).total_seconds() / 3600.0 for t in timestamps])


def _sample_sufficiency_confidence(n: int, required: int = MIN_SAMPLES) -> float:
    return max(0.0, min(1.0, n / required))


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def compute_forecast(readings: list[tuple[datetime, float]], nominal: float,
                      spec_low: float, spec_high: float, noise_sd: float,
                      sensor_tier: str, station_id: str, signal: str) -> DegradationForecast:
    n_raw = len(readings)

    if n_raw < MIN_SAMPLES:
        return DegradationForecast(
            station_id=station_id, signal=signal,
            current_value=readings[-1][1] if readings else nominal,
            slope_per_hour=0.0, slope_ci=(0.0, 0.0), spec_limit=None,
            hours_to_limit=None, hours_to_limit_ci=None,
            confidence=_sample_sufficiency_confidence(n_raw) * TIER_CONFIDENCE.get(sensor_tier, 0.3),
            status=STATUS_UNSTABLE,
            reason=f"insufficient samples (n={n_raw}, need >= {MIN_SAMPLES})",
            n_samples=n_raw, window_start_ts=readings[0][0] if readings else None,
        )

    timestamps = [r[0] for r in readings]
    values = np.array([r[1] for r in readings], dtype=float)

    # --- Step 2: changepoint reset ---
    cp_idx = detect_changepoint(values)
    if cp_idx is not None and (n_raw - cp_idx) <= CHANGEPOINT_RECENT_SAMPLES:
        return DegradationForecast(
            station_id=station_id, signal=signal, current_value=float(values[-1]),
            slope_per_hour=0.0, slope_ci=(0.0, 0.0), spec_limit=None,
            hours_to_limit=None, hours_to_limit_ci=None,
            confidence=TIER_CONFIDENCE.get(sensor_tier, 0.3) * 0.5,
            status=STATUS_UNSTABLE,
            reason="step change detected — not a wear trend",
            n_samples=n_raw, window_start_ts=timestamps[0],
        )
    if cp_idx is not None:
        timestamps = timestamps[cp_idx:]
        values = values[cp_idx:]

    n = len(values)
    if n < MIN_SAMPLES:
        return DegradationForecast(
            station_id=station_id, signal=signal, current_value=float(values[-1]),
            slope_per_hour=0.0, slope_ci=(0.0, 0.0), spec_limit=None,
            hours_to_limit=None, hours_to_limit_ci=None,
            confidence=_sample_sufficiency_confidence(n) * TIER_CONFIDENCE.get(sensor_tier, 0.3),
            status=STATUS_UNSTABLE,
            reason=f"insufficient samples post-changepoint (n={n}, need >= {MIN_SAMPLES})",
            n_samples=n, window_start_ts=timestamps[0],
        )

    elapsed_h = _elapsed_hours(timestamps)
    current_value = float(values[-1])

    # --- Step 4: robust slope (Theil-Sen) ---
    # Fit on the RAW readings, not the EWMA-smoothed series. EWMA output is
    # serially correlated by construction; feeding it to Theil-Sen violates the
    # estimator's independence assumption and artificially narrows its CI, which
    # inflates the false-positive ("trending") rate on pure noise — the opposite
    # of what step 5's significance gate is supposed to guarantee. Fitting on raw
    # values keeps the CI honest; the EWMA is used only to report a denoised
    # "current level" for the projection below. Deviation from a literal reading
    # of §6.2 step 4, documented in ASSUMPTIONS.md.
    # alpha=0.99, not the textbook 0.95: this gate is evaluated on every tick against
    # a heavily overlapping (highly autocorrelated) rolling window, not as one
    # independent trial — a 95% CI lets spurious "significant" runs of ticks through
    # far more than 5% of the time in that setting. 0.99 was tuned empirically against
    # a pure-noise signal to bring the false-forecast rate under the §6.4 target of
    # <0.5 episodes/shift; see validation/backtest.py for the measured rate.
    slope, intercept, lo_slope, up_slope = stats.theilslopes(values, elapsed_h, alpha=0.99)
    current_smoothed = float(intercept + slope * elapsed_h[-1])

    base_reason = ""

    # --- Step 5: significance gate ---
    if lo_slope <= 0 <= up_slope:
        return DegradationForecast(
            station_id=station_id, signal=signal, current_value=current_value,
            slope_per_hour=float(slope), slope_ci=(float(lo_slope), float(up_slope)),
            spec_limit=None, hours_to_limit=None, hours_to_limit_ci=None,
            confidence=TIER_CONFIDENCE.get(sensor_tier, 0.3),
            status=STATUS_STABLE,
            reason="slope CI includes zero — not significantly trending",
            n_samples=n, window_start_ts=timestamps[0],
        )

    # --- Step 7a: residual spread sanity gate ---
    residuals = values - (intercept + slope * elapsed_h)
    residual_std = float(np.std(residuals))
    if residual_std > RESIDUAL_SPREAD_FACTOR * max(noise_sd, 1e-9):
        return DegradationForecast(
            station_id=station_id, signal=signal, current_value=current_value,
            slope_per_hour=float(slope), slope_ci=(float(lo_slope), float(up_slope)),
            spec_limit=None, hours_to_limit=None, hours_to_limit_ci=None,
            confidence=TIER_CONFIDENCE.get(sensor_tier, 0.3) * 0.4,
            status=STATUS_UNSTABLE,
            reason="residual spread exceeds threshold — linear fit does not describe the data",
            n_samples=n, window_start_ts=timestamps[0],
        )

    # --- Step 6: project to limit ---
    spec_limit = spec_high if slope > 0 else spec_low
    if (slope > 0 and current_smoothed >= spec_limit) or (slope < 0 and current_smoothed <= spec_limit):
        return DegradationForecast(
            station_id=station_id, signal=signal, current_value=current_value,
            slope_per_hour=float(slope), slope_ci=(float(lo_slope), float(up_slope)),
            spec_limit=float(spec_limit), hours_to_limit=0.0, hours_to_limit_ci=(0.0, 0.0),
            confidence=TIER_CONFIDENCE.get(sensor_tier, 0.3),
            status=STATUS_CROSSED,
            reason="already at or beyond spec limit",
            n_samples=n, window_start_ts=timestamps[0],
        )

    candidates = []
    for s in (slope, lo_slope, up_slope):
        if s == 0 or (s > 0) != (slope > 0):
            continue
        candidates.append((spec_limit - current_smoothed) / s)
    hours_to_limit = (spec_limit - current_smoothed) / slope
    hours_ci = (min(candidates), max(candidates)) if candidates else (hours_to_limit, hours_to_limit)

    # The slope CI alone only propagates uncertainty in the trend's steepness; it
    # says nothing about uncertainty in *where the line currently sits* (the
    # intercept), so as the crossing nears and slope-driven uncertainty shrinks
    # toward zero, the interval collapses even though there is still real residual
    # uncertainty about the current level. Add the standard regression
    # prediction-interval term for the value at the current point (leverage-scaled
    # residual spread), converted into an hours-margin via the slope — without
    # this term, measured 95%-CI coverage against ground truth was ~64%, well
    # short of the honest ~95% BUILD_SPEC.md §6.4 asks for.
    if abs(slope) > 1e-9:
        mean_h = float(elapsed_h.mean())
        sxx = float(np.sum((elapsed_h - mean_h) ** 2)) + 1e-9
        leverage = 1.0 / n + (elapsed_h[-1] - mean_h) ** 2 / sxx
        z = 2.576  # matches the 0.99 CI level used for the slope fit above
        value_margin = residual_std * np.sqrt(leverage) * z
        extra_h = value_margin / abs(slope)
        hours_ci = (hours_ci[0] - extra_h, hours_ci[1] + extra_h)

    # --- Step 8: curvature check ---
    if n >= 2 * CURVATURE_MIN_HALF_SAMPLES:
        mid = n // 2
        h1, v1 = elapsed_h[:mid], values[:mid]
        h2, v2 = elapsed_h[mid:], values[mid:]
        try:
            slope1, intercept1 = stats.theilslopes(v1, h1)[:2]
            slope2, intercept2 = stats.theilslopes(v2, h2)[:2]
        except Exception:
            slope1, intercept1, slope2, intercept2 = slope, intercept, slope, intercept
        same_sign = (slope1 >= 0) == (slope >= 0) and (slope2 >= 0) == (slope >= 0)
        if same_sign and abs(slope1) > 1e-9 and abs(slope2) > abs(slope1) * CURVATURE_FACTOR:
            # A locally-linear extrapolation from just the second half still
            # under-predicts a genuinely accelerating process — the curve keeps
            # steepening past even the recent local rate. Measured against
            # ground truth, that alone left forecasts 3-8h optimistic at 5-7h
            # lead time on real accelerating_wear episodes: the dangerous
            # direction (says "more time than you have"). Instead estimate the
            # curvature explicitly (rate of change of slope between the two
            # half-windows) and extrapolate the resulting quadratic from "now",
            # the same closed form simulator/degradation.py uses for ground truth.
            mid1_h, mid2_h = float(h1.mean()), float(h2.mean())
            accel_est = (slope2 - slope1) / max(mid2_h - mid1_h, 1e-9)
            t_now = float(elapsed_h[-1])
            v_now = intercept2 + slope2 * t_now
            a_coef, b_coef, c_coef = 0.5 * accel_est, slope2, v_now - spec_limit
            quad_hours = None
            if abs(a_coef) < 1e-9:
                if abs(b_coef) > 1e-9 and -c_coef / b_coef > 0:
                    quad_hours = -c_coef / b_coef
            else:
                disc = b_coef * b_coef - 4 * a_coef * c_coef
                if disc >= 0:
                    sq = disc ** 0.5
                    positive = [r for r in ((-b_coef + sq) / (2 * a_coef), (-b_coef - sq) / (2 * a_coef)) if r > 0]
                    if positive:
                        quad_hours = min(positive)
            if quad_hours is not None:
                hours_to_limit = quad_hours
                # Merging quad_hours into the linear-fit CI via min/max alone can
                # produce a one-sided interval with no margin on the far side of
                # the quadratic point (the true crossing could land on either
                # side of it) — reuse the already-computed value-uncertainty
                # margin as a symmetric band around quad_hours too.
                margin = extra_h if abs(slope) > 1e-9 else 0.0
                hours_ci = (min(hours_ci[0], quad_hours - margin), max(hours_ci[1], quad_hours + margin))
            base_reason = "accelerating — quadratic extrapolation from recent curvature; "

    # --- Step 7b: planning horizon gate ---
    if hours_to_limit > MAX_HORIZON_HOURS:
        return DegradationForecast(
            station_id=station_id, signal=signal, current_value=current_value,
            slope_per_hour=float(slope), slope_ci=(float(lo_slope), float(up_slope)),
            spec_limit=float(spec_limit), hours_to_limit=None, hours_to_limit_ci=None,
            confidence=TIER_CONFIDENCE.get(sensor_tier, 0.3),
            status=STATUS_DRIFTING,
            reason=base_reason + "drifting slowly — projected crossing is beyond the 72h planning horizon",
            n_samples=n, window_start_ts=timestamps[0],
        )

    n_conf = _sample_sufficiency_confidence(n)
    confidence = TIER_CONFIDENCE.get(sensor_tier, 0.3) * (0.5 + 0.5 * n_conf)

    return DegradationForecast(
        station_id=station_id, signal=signal, current_value=current_value,
        slope_per_hour=float(slope), slope_ci=(float(lo_slope), float(up_slope)),
        spec_limit=float(spec_limit), hours_to_limit=float(hours_to_limit),
        hours_to_limit_ci=(float(hours_ci[0]), float(hours_ci[1])),
        confidence=confidence, status=STATUS_FORECASTABLE,
        reason=base_reason + f"drifting at {slope:+.4g}/h toward {spec_limit:g}",
        n_samples=n, window_start_ts=timestamps[0],
    )


# ---------------------------------------------------------------------------
# Stateful forecaster over a whole line
# ---------------------------------------------------------------------------

class Forecaster:
    """Stateful, per-signal wrapper around compute_forecast().

    Adds a persistence requirement on top of the single-window significance gate:
    consecutive rolling windows over autocorrelated noise are NOT independent
    trials, so a fixed-alpha CI alone lets far more than alpha's share of spurious
    "trending" windows through over a shift. Requiring PERSISTENCE_TICKS
    consecutive significant, same-direction evaluations before surfacing a
    forecast is a standard monitoring-system debounce; it costs a few minutes of
    lead time and materially cuts the false-forecast rate (see
    validation/backtest.py for the measured number against the <0.5/shift target
    in BUILD_SPEC.md §6.4).
    """

    def __init__(self, station_configs):
        self.specs = {
            (s.id, sig): spec for s in station_configs for sig, spec in (s.signals or {}).items()
        }
        self.tiers = {s.id: s.sensor_tier for s in station_configs}
        self.buffers: dict[tuple, SignalBuffer] = {}
        self._streak: dict[tuple, int] = {}
        self._streak_sign: dict[tuple, int] = {}

    def update(self, station_id: str, signal: str, ts: datetime, value: float) -> DegradationForecast:
        key = (station_id, signal)
        buf = self.buffers.setdefault(key, SignalBuffer())
        buf.add(ts, value)
        spec = self.specs[key]
        fc = compute_forecast(
            buf.readings, spec["nominal"], spec["spec_low"], spec["spec_high"], spec["noise_sd"],
            self.tiers[station_id], station_id, signal,
        )

        if fc.status in (STATUS_FORECASTABLE, STATUS_DRIFTING, STATUS_CROSSED):
            sign = 1 if fc.slope_per_hour >= 0 else -1
            if self._streak_sign.get(key) == sign:
                self._streak[key] = self._streak.get(key, 0) + 1
            else:
                self._streak[key] = 1
                self._streak_sign[key] = sign
            if self._streak[key] < PERSISTENCE_TICKS and fc.status != STATUS_CROSSED:
                fc = DegradationForecast(
                    station_id=fc.station_id, signal=fc.signal, current_value=fc.current_value,
                    slope_per_hour=fc.slope_per_hour, slope_ci=fc.slope_ci, spec_limit=None,
                    hours_to_limit=None, hours_to_limit_ci=None, confidence=fc.confidence,
                    status=STATUS_STABLE,
                    reason=f"trend detected but not yet persistent ({self._streak[key]}/{PERSISTENCE_TICKS} consecutive)",
                    n_samples=fc.n_samples, window_start_ts=fc.window_start_ts,
                )
        else:
            self._streak[key] = 0
            self._streak_sign[key] = 0

        return fc

    def instrumented_pairs(self) -> list[tuple[str, str]]:
        return list(self.specs.keys())
