"""Live line state as of a snapshot timestamp — the single source all three
dashboard tabs read from ("one model, three views", BUILD_SPEC.md §10).

A real deployment would feed this from a live event stream; here it replays
pre-generated scenario data up to `as_of`, which is what makes the dashboard's
time slider work — nothing after `as_of` is ever visible to any model, which is
what makes the "we flagged it 78 minutes before EOL testing would have" claim
honest rather than hindsight.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from models.bottleneck import bottleneck_hours_by_station, bottleneck_migration
from models.detect import Detection, Detector
from models.forecast import DegradationForecast, Forecaster, STATUS_STABLE
from models.infer import AtRiskVehicle, at_risk_table, infer_at_risk_vehicles
from simulator.line import LineConfig
from twin.confidence import TIER_CONFIDENCE
from twin.genealogy import Genealogy

DEFAULT_LOOKBACK_HOURS = 24.0  # bounds live replay cost on multi-day scenarios


@dataclass
class TwinState:
    as_of: datetime
    line_cfg: LineConfig
    events: pd.DataFrame          # full history up to as_of (for genealogy/ETA)
    status: pd.DataFrame
    genealogy: Genealogy
    detections: list              # Detection, most recent lookback window only
    forecasts: dict                # (station_id, signal) -> DegradationForecast (latest)
    at_risk: pd.DataFrame
    zone_of: dict = field(default_factory=dict)
    tier_of: dict = field(default_factory=dict)


def build_twin_state(line_cfg: LineConfig, events_df: pd.DataFrame, status_df: pd.DataFrame,
                      as_of: datetime, lookback_hours: float = DEFAULT_LOOKBACK_HOURS) -> TwinState:
    live_events = events_df[events_df.exit_ts <= as_of]
    live_status = status_df[status_df.start_ts <= as_of]
    genealogy = Genealogy(live_events, line_cfg)

    zone_of = {s.id: s.zone for s in line_cfg.stations}
    tier_of = {s.id: s.sensor_tier for s in line_cfg.stations}

    replay_start = as_of - timedelta(hours=lookback_hours)
    detector = Detector(line_cfg.stations)
    forecaster = Forecaster(line_cfg.stations)

    detections: list[Detection] = []
    forecasts: dict[tuple, DegradationForecast] = {}
    for station_id, signal in forecaster.instrumented_pairs():
        if signal not in live_events.columns:
            continue
        sub = live_events[
            (live_events.station_id == station_id) & live_events[signal].notna()
            & (live_events.entry_ts >= replay_start)
        ].sort_values("entry_ts")
        for row in sub.itertuples():
            value = getattr(row, signal)
            detections.extend(detector.update(station_id, signal, value, row.entry_ts))
            forecasts[(station_id, signal)] = forecaster.update(station_id, signal, row.entry_ts, value)

    at_risk_rows: list[AtRiskVehicle] = []
    for (station_id, signal), fc in forecasts.items():
        if fc.status == STATUS_STABLE:
            continue
        deviation_pct = abs(fc.current_value - (fc.spec_limit or fc.current_value)) / max(abs(fc.current_value), 1e-9) * 100
        window_start = fc.window_start_ts or replay_start
        vehicles = infer_at_risk_vehicles(
            genealogy, station_id, signal, window_start, as_of,
            deviation_pct=deviation_pct, zone_of=zone_of,
            sensor_tier=tier_of.get(station_id, "C"), tier_confidence=TIER_CONFIDENCE,
        )
        at_risk_rows.extend(vehicles)

    return TwinState(
        as_of=as_of, line_cfg=line_cfg, events=live_events, status=live_status,
        genealogy=genealogy, detections=detections, forecasts=forecasts,
        at_risk=at_risk_table(at_risk_rows), zone_of=zone_of, tier_of=tier_of,
    )


def current_station_status(status_df: pd.DataFrame, as_of: datetime) -> dict:
    """Last known status per station at `as_of` (for the supervisor line map)."""
    window = status_df[(status_df.start_ts <= as_of) & (status_df.end_ts >= as_of)]
    if window.empty:
        window = status_df[status_df.start_ts <= as_of].sort_values("start_ts")
        window = window.groupby("station_id").tail(1)
    return dict(zip(window.station_id, window.status))
