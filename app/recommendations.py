"""Recommendation objects with an enforced status lifecycle (BUILD_SPEC.md §2.1,
hard constraint 1): the twin is read-only. It emits recommendations; a human
executes them. There is no "Auto-apply" anywhere in this codebase.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

STATUS_PROPOSED = "proposed"
STATUS_ACKNOWLEDGED = "acknowledged"
STATUS_EXECUTED = "executed_by_human"

VALID_TRANSITIONS = {
    STATUS_PROPOSED: {STATUS_ACKNOWLEDGED},
    STATUS_ACKNOWLEDGED: {STATUS_EXECUTED},
    STATUS_EXECUTED: set(),
}

SHIFT_LENGTH_H = 8


@dataclass
class Recommendation:
    id: str
    title: str
    detail: str
    station_id: str
    status: str = STATUS_PROPOSED

    def advance(self) -> None:
        next_status = {
            STATUS_PROPOSED: STATUS_ACKNOWLEDGED,
            STATUS_ACKNOWLEDGED: STATUS_EXECUTED,
        }.get(self.status)
        if next_status is None:
            raise ValueError(f"cannot advance a recommendation already {self.status!r}")
        self.status = next_status


def next_shift_change(ts: datetime) -> datetime:
    minutes_into_shift = (ts.hour * 60 + ts.minute) % (SHIFT_LENGTH_H * 60)
    minutes_remaining = SHIFT_LENGTH_H * 60 - minutes_into_shift
    return (ts.replace(second=0, microsecond=0) + timedelta(minutes=minutes_remaining))


def forecast_recommendation(station_id: str, signal: str, forecast, as_of: datetime) -> Recommendation:
    rec_id = f"{station_id}:{signal}:{as_of.isoformat()}"
    if forecast.hours_to_limit is not None:
        service_by = next_shift_change(as_of)
        title = f"{station_id} {signal} — {forecast.hours_to_limit:.1f}h to tolerance limit"
        detail = (
            f"Drifting at {forecast.slope_per_hour:+.3g}/h toward {forecast.spec_limit:g}. "
            f"Range {forecast.hours_to_limit_ci[0]:.1f}-{forecast.hours_to_limit_ci[1]:.1f}h. "
            f"Recommend scheduling service at the {service_by.strftime('%H:%M')} shift change."
        )
    else:
        title = f"{station_id} {signal} — {forecast.status}"
        detail = forecast.reason
    return Recommendation(id=rec_id, title=title, detail=detail, station_id=station_id)


def _forecaster_progress_line(forecast) -> str:
    """Explains *why* no forecast exists yet, so the supervisor isn't left
    silently waiting with no idea whether the twin is still gathering data,
    found no trend, or rejected the signal as unstable."""
    if forecast is None:
        return "Forecaster has no data yet for this signal."
    if forecast.status == "unstable" and forecast.reason.startswith("insufficient samples"):
        # reason is e.g. "insufficient samples (n=34, need >= 60)" — surface it directly
        return f"Forecaster status: gathering samples ({forecast.reason.split('(')[-1].rstrip(')')})."
    if forecast.status == "stable":
        return "Forecaster status: no significant trend yet — could still be a one-off, not sustained wear."
    if forecast.status == "unstable":
        return f"Forecaster status: {forecast.reason}."
    return f"Forecaster status: {forecast.status} — {forecast.reason}"


def detection_recommendation(station_id: str, signal: str, detection, as_of: datetime, forecast=None) -> Recommendation:
    """A detection alone — before the forecaster has accumulated enough
    consecutive significant readings to commit to a countdown — is still
    something a supervisor should look at now. This is the "investigate now"
    card; forecast_recommendation() supersedes it once a forecast exists.

    `forecast`, if given, is the current (possibly stable/unstable)
    DegradationForecast for this signal — shown as a progress line so there's
    never a silent wait with no visibility into what the forecaster is doing.
    """
    rec_id = f"{station_id}:{signal}:{as_of.isoformat()}"
    title = f"{station_id} {signal} — active detection, no forecast yet"
    detail = (
        f"{detection.reason} (value={detection.value:.2f}, severity={detection.severity}). "
        f"{_forecaster_progress_line(forecast)}"
    )
    return Recommendation(id=rec_id, title=title, detail=detail, station_id=station_id)
