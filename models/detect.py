"""Spec-limit detection + EWMA control charts (BUILD_SPEC.md §7.1).

Zero training data required — this is the credibility floor plant engineers
already trust. Detection reports what IS happening now; never confused with
forecasting (what WILL happen) or inference (what already happened, unseen).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

EWMA_ALPHA = 0.2
CONTROL_LIMIT_SIGMAS = 3.0  # standard EWMA control chart width

SEVERITY_INFO = "info"
SEVERITY_WARNING = "warning"
SEVERITY_CRITICAL = "critical"


@dataclass
class Detection:
    station_id: str
    signal: str
    value: float
    nominal: float
    deviation_pct: float
    severity: str
    reason: str
    ts: datetime
    confirmed: bool = True
    """False for a single-tick EWMA breach that hasn't yet persisted long enough
    to rule out a rare, ordinary statistical fluctuation (see EWMAControlChart).
    Still surfaced — never silently dropped — but must not, on its own, drive a
    recommendation card: an operator seeing "investigate now" on every isolated
    blip is exactly the alarm fatigue this system is trying to avoid. Spec-limit
    breaches are always confirmed=True: a hard physical spec violation on a
    single reading is a fact, not a statistical inference needing persistence."""


def spec_limit_check(station_id: str, signal: str, value: float, ts: datetime,
                      nominal: float, spec_low: float, spec_high: float) -> Optional[Detection]:
    deviation_pct = (value - nominal) / nominal * 100.0 if nominal else 0.0
    if value < spec_low or value > spec_high:
        return Detection(station_id, signal, value, nominal, deviation_pct,
                          SEVERITY_CRITICAL, "out of spec limits", ts, confirmed=True)
    span = spec_high - spec_low
    band = span * 0.1
    if value > spec_high - band or value < spec_low + band:
        return Detection(station_id, signal, value, nominal, deviation_pct,
                          SEVERITY_WARNING, "within marginal band of spec limit", ts, confirmed=True)
    return None


CONSECUTIVE_BREACHES_REQUIRED = 2  # standard SPC "runs rule" — see note below


class EWMAControlChart:
    """Standard EWMA control chart with asymptotic control limits.

    center = nominal (process target); flags when the EWMA statistic exceeds
    the standard control-limit band, independent of hard spec limits — this
    catches drift that hasn't yet reached spec but is unusual for the process.

    Flags on the FIRST breach — a real spike happened, and hiding that from the
    operator would be its own kind of dishonesty — but marks it
    confirmed=False until CONSECUTIVE_BREACHES_REQUIRED same-direction
    breaches have been seen. A fixed 3-sigma chart has a small but real
    single-point false-alarm rate, and run across many signals over many
    thousands of ticks that adds up to real false alarms (measured: 2 isolated
    single-tick breaches on stations with zero scheduled anomaly across the
    `weld_drift_demo` run). Real SPC practice handles exactly this with a "runs
    rule" (e.g. 2-of-2 or 2-of-3 consecutive points beyond the limit) — but the
    rule should gate what's *actionable*, not what's *visible*. Only confirmed
    detections may drive a user-facing "investigate now" recommendation card;
    unconfirmed ones still show up in the detections list, explicitly labelled,
    so a real (if isolated) spike is never silently dropped, and an operator
    isn't left guessing why nothing appeared.
    """

    def __init__(self, nominal: float, noise_sd: float, alpha: float = EWMA_ALPHA,
                 limit_sigmas: float = CONTROL_LIMIT_SIGMAS):
        self.nominal = nominal
        self.noise_sd = noise_sd
        self.alpha = alpha
        self.limit_sigmas = limit_sigmas
        self.ewma = nominal
        self.n = 0
        self._breach_streak = 0
        self._breach_sign = 0

    def update(self, station_id: str, signal: str, value: float, ts: datetime) -> Optional[Detection]:
        self.n += 1
        self.ewma = self.alpha * value + (1 - self.alpha) * self.ewma
        # asymptotic EWMA control-limit std (converges as n grows)
        factor = (self.alpha / (2 - self.alpha)) * (1 - (1 - self.alpha) ** (2 * self.n))
        ewma_sd = self.noise_sd * (factor ** 0.5)
        limit = self.limit_sigmas * max(ewma_sd, 1e-9)
        deviation = self.ewma - self.nominal

        if abs(deviation) <= limit:
            self._breach_streak = 0
            self._breach_sign = 0
            return None

        sign = 1 if deviation > 0 else -1
        self._breach_streak = self._breach_streak + 1 if sign == self._breach_sign else 1
        self._breach_sign = sign

        confirmed = self._breach_streak >= CONSECUTIVE_BREACHES_REQUIRED
        deviation_pct = deviation / self.nominal * 100.0 if self.nominal else 0.0
        severity = SEVERITY_CRITICAL if abs(deviation) > 1.5 * limit else SEVERITY_WARNING
        reason = (
            f"EWMA chart breaks {self.limit_sigmas:g}-sigma control limit ({self._breach_streak} consecutive)"
            if confirmed else
            f"EWMA chart breaks {self.limit_sigmas:g}-sigma control limit — single tick, "
            f"unconfirmed (needs {CONSECUTIVE_BREACHES_REQUIRED} consecutive; possible false alarm)"
        )
        return Detection(station_id, signal, value, self.nominal, deviation_pct,
                          severity, reason, ts, confirmed=confirmed)


class Detector:
    """Runs both spec-limit checks and EWMA control charts across all
    instrumented (station, signal) pairs."""

    def __init__(self, station_configs):
        self.specs = {
            (s.id, sig): spec for s in station_configs for sig, spec in (s.signals or {}).items()
        }
        self.charts: dict[tuple, EWMAControlChart] = {}

    def update(self, station_id: str, signal: str, value: float, ts: datetime) -> list[Detection]:
        key = (station_id, signal)
        spec = self.specs[key]
        out = []
        spec_hit = spec_limit_check(station_id, signal, value, ts,
                                     spec["nominal"], spec["spec_low"], spec["spec_high"])
        if spec_hit:
            out.append(spec_hit)
        chart = self.charts.setdefault(key, EWMAControlChart(spec["nominal"], spec["noise_sd"]))
        chart_hit = chart.update(station_id, signal, value, ts)
        if chart_hit:
            out.append(chart_hit)
        return out
