"""Latent defect injection and delayed surfacing at inspection stations.

Implements the DefectInjector hook from simulator/line.py: at the moment a
vehicle finishes processing at a station, decide whether it picks up a latent
defect, based on how far the *actual* (degradation-driven) signal mean is from
spec at that instant — including the marginal-band case from BUILD_SPEC.md §5.4.

Surfacing (when the defect is actually discovered) is NOT decided live — it
depends on the vehicle's full downstream station history, so it is resolved
after the run by surface_defects(), which walks each defect's VIN forward to
its designated inspection station (S28 body geometry, S38 paint finish,
S41 leak/electrical EOL).
"""
from __future__ import annotations

import random
from typing import Optional

import pandas as pd

from simulator.line import DefectInjector, StationConfig
from simulator.degradation import DegradationEngine

# Explicit overrides for the tier-A/B signals that carry a specific narrative
# (BUILD_SPEC.md §5.4/§14: a weld defect at S12 shows up as a leak/electrical
# failure at end-of-line, not as a body-geometry defect). Documented in
# ASSUMPTIONS.md. Anything not listed here falls back to a zone-based default.
DEFECT_SURFACE_OVERRIDES = {
    ("S12", "weld_current_a"): "S41",
    ("S19", "paint_flow_rate_mlmin"): "S38",
    ("S31", "press_force_kn"): "S28",
}

ZONE_DEFAULT_INSPECTION = {
    "body": "S28",
    "paint": "S38",
    "assembly": "S41",
}


class LatentDefectInjector(DefectInjector):
    def __init__(self, station_configs: list[StationConfig], degradation_engine: DegradationEngine,
                 seed: int = 7, marginal_band_frac: float = 0.10,
                 prob_out_of_spec: float = 0.35, prob_marginal: float = 0.08,
                 prob_background: float = 0.001):
        self.engine = degradation_engine
        self.rng2 = random.Random(seed)
        self.station_signals = {s.id: (s.signals or {}) for s in station_configs}
        self.marginal_band_frac = marginal_band_frac
        self.prob_out_of_spec = prob_out_of_spec
        self.prob_marginal = prob_marginal
        self.prob_background = prob_background

    def maybe_injure(self, station_id: str, vin: str, variant: str, ts, rng: random.Random) -> Optional[dict]:
        signals = self.station_signals.get(station_id)
        if not signals:
            # Tier-C (dark) station: no signal to key off, low background rate only.
            if self.rng2.random() < self.prob_background:
                return {"defect_type": "unspecified", "root_cause": f"{station_id}:background", "signal": None}
            return None

        for sig_name, spec in signals.items():
            mean_val = self.engine.mean_value_at_ts(station_id, sig_name, ts)
            spec_low, spec_high = spec["spec_low"], spec["spec_high"]
            span = spec_high - spec_low
            band = span * self.marginal_band_frac

            if mean_val > spec_high or mean_val < spec_low:
                prob, cause = self.prob_out_of_spec, "out_of_spec"
            elif mean_val > spec_high - band or mean_val < spec_low + band:
                prob, cause = self.prob_marginal, "marginal_band"
            else:
                prob, cause = self.prob_background, "background"

            if self.rng2.random() < prob:
                return {
                    "defect_type": f"{sig_name}_deviation",
                    "root_cause": f"{station_id}:{sig_name}:{cause}",
                    "signal": sig_name,
                }
        return None


def _target_inspection_station(injected_station: str, signal: Optional[str],
                                zone_of: dict) -> str:
    if signal is not None and (injected_station, signal) in DEFECT_SURFACE_OVERRIDES:
        return DEFECT_SURFACE_OVERRIDES[(injected_station, signal)]
    zone = zone_of.get(injected_station, "assembly")
    return ZONE_DEFAULT_INSPECTION.get(zone, "S41")


def surface_defects(events_df: pd.DataFrame, defects_df: pd.DataFrame,
                     station_configs: list[StationConfig],
                     detection_prob: float = 0.9, seed: int = 99) -> pd.DataFrame:
    """Post-process ground-truth defects against the full events log to resolve
    when/whether each one is actually caught at its designated inspection station.

    Adds/overwrites: surfaced_at_station, surfaced_ts, reworked.
    A defect can escape (reworked=False even though it reached inspection) with
    probability (1 - detection_prob), modelling imperfect inspection.
    """
    if defects_df.empty:
        return defects_df.assign(surfaced_at_station=None, surfaced_ts=None, reworked=False)

    rng = random.Random(seed)
    zone_of = {s.id: s.zone for s in station_configs}
    events_sorted = events_df.sort_values(["vin", "entry_ts"])
    by_vin = {vin: g for vin, g in events_sorted.groupby("vin")}

    records = []
    for _, d in defects_df.iterrows():
        rec = dict(d)
        target_station = _target_inspection_station(d["injected_at_station"], d.get("signal"), zone_of)
        rec["surfaced_at_station"] = None
        rec["surfaced_ts"] = None
        rec["reworked"] = False

        g = by_vin.get(d["vin"])
        if g is not None:
            after = g[(g.entry_ts >= d["injected_ts"]) & (g.station_id == target_station)]
            if len(after):
                visit = after.iloc[0]
                rec["surfaced_at_station"] = target_station
                rec["surfaced_ts"] = visit["exit_ts"]
                rec["reworked"] = bool(rng.random() < detection_prob)
        records.append(rec)

    return pd.DataFrame(records)
