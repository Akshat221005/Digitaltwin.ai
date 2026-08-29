"""Latent defect inference (BUILD_SPEC.md §7.3) — the "63 vehicles are already
carrying this" query. Given a detection or forecast window at station X, finds
every VIN that passed X during the window and hasn't yet reached its inspection
station, scores each one's risk, and reports an expected surfacing ETA.

Inference is not detection (nothing is wrong with the VIN *right now* that a
scan would show) and not forecasting (nothing is projected forward in time) —
it is a claim about a latent condition that already exists, unseen.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd

from twin.genealogy import Genealogy

# Mirrors simulator/defects.py's mapping: which inspection station would catch
# a defect that originated at a given zone / signal. Kept in sync manually since
# inference must reason about this without access to ground truth.
from simulator.defects import DEFECT_SURFACE_OVERRIDES, ZONE_DEFAULT_INSPECTION


@dataclass
class AtRiskVehicle:
    vin: str
    variant: str
    origin_station: str
    origin_entry_ts: datetime
    last_known_station: str
    last_known_ts: datetime
    expected_surfacing_station: str
    eta: Optional[datetime]
    risk_score: float
    confidence: float


def _target_inspection_station(origin_station: str, signal: Optional[str], zone_of: dict) -> str:
    if signal is not None and (origin_station, signal) in DEFECT_SURFACE_OVERRIDES:
        return DEFECT_SURFACE_OVERRIDES[(origin_station, signal)]
    zone = zone_of.get(origin_station, "assembly")
    return ZONE_DEFAULT_INSPECTION.get(zone, "S41")


def infer_at_risk_vehicles(genealogy: Genealogy, station_id: str, signal: Optional[str],
                            window_start: datetime, window_end: datetime,
                            deviation_pct: float, zone_of: dict,
                            sensor_tier: str, tier_confidence: dict) -> list[AtRiskVehicle]:
    """Given a detection/forecast at (station_id, signal) active over
    [window_start, window_end), returns every VIN that passed the station during
    that window and hasn't yet reached its expected inspection station."""
    target_station = _target_inspection_station(station_id, signal, zone_of)
    in_transit = genealogy.vins_in_transit_to(station_id, target_station, window_start, window_end)
    if in_transit.empty:
        return []

    results = []
    for row in in_transit.itertuples():
        eta = genealogy.eta_to_station(row.vin, target_station)
        dwell_anomaly = 0.0  # placeholder hook: could compare row's dwell to station baseline
        time_in_window_frac = 1.0  # every returned VIN passed fully within the window by construction
        risk_score = _risk_score(deviation_pct, dwell_anomaly, time_in_window_frac)
        confidence = tier_confidence.get(sensor_tier, 0.3)
        results.append(AtRiskVehicle(
            vin=row.vin, variant=row.variant, origin_station=station_id,
            origin_entry_ts=row.origin_entry_ts, last_known_station=row.last_station,
            last_known_ts=row.last_ts, expected_surfacing_station=target_station,
            eta=eta, risk_score=risk_score, confidence=confidence,
        ))
    return results


def _risk_score(deviation_pct: float, dwell_anomaly: float, time_in_window_frac: float) -> float:
    """0..1 risk score from deviation magnitude, dwell anomaly, and time-in-window.
    Weights are a simple, defensible starting point — deviation dominates since
    it's the most direct evidence, per BUILD_SPEC.md §7.3."""
    dev_component = min(1.0, abs(deviation_pct) / 20.0)  # 20% deviation -> saturates at 1.0
    dwell_component = min(1.0, abs(dwell_anomaly) / 3.0)  # 3 sigma dwell anomaly -> saturates
    score = 0.65 * dev_component + 0.20 * dwell_component + 0.15 * time_in_window_frac
    return max(0.0, min(1.0, score))


def at_risk_table(vehicles: list[AtRiskVehicle]) -> pd.DataFrame:
    if not vehicles:
        return pd.DataFrame(columns=[
            "vin", "variant", "origin_station", "origin_entry_ts", "last_known_station",
            "last_known_ts", "expected_surfacing_station", "eta", "risk_score", "confidence",
        ])
    return pd.DataFrame([vars(v) for v in vehicles]).sort_values("risk_score", ascending=False)
