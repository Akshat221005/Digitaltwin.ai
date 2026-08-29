"""Per-vehicle station history, reconstructed from events.parquet.

This is the substrate models/infer.py queries to answer "which VINs passed
station X during the drift window and have not yet reached their inspection
station" — the "63 vehicles are already carrying this" query.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from simulator.line import LineConfig


class Genealogy:
    def __init__(self, events_df: pd.DataFrame, line_cfg: LineConfig):
        self.events = events_df.sort_values(["vin", "entry_ts"]).reset_index(drop=True)
        self.station_order = [s.id for s in line_cfg.stations]
        self._order_index = {sid: i for i, sid in enumerate(self.station_order)}
        self._transit_cache: dict[tuple[str, str], float] = {}

    # ------------------------------------------------------------------
    # Per-VIN history
    # ------------------------------------------------------------------

    def vin_history(self, vin: str) -> pd.DataFrame:
        return self.events[self.events.vin == vin]

    def last_known_station(self, vin: str) -> Optional[tuple[str, pd.Timestamp]]:
        hist = self.vin_history(vin)
        if hist.empty:
            return None
        last = hist.iloc[-1]
        return last["station_id"], last["exit_ts"]

    def has_reached(self, vin: str, station_id: str) -> bool:
        hist = self.vin_history(vin)
        return bool((hist.station_id == station_id).any())

    def station_of_entry(self, vin: str, station_id: str) -> Optional[pd.Timestamp]:
        hist = self.vin_history(vin)
        row = hist[hist.station_id == station_id]
        return row.iloc[0]["entry_ts"] if len(row) else None

    # ------------------------------------------------------------------
    # Cross-VIN queries
    # ------------------------------------------------------------------

    def station_window(self, station_id: str, start_ts: datetime, end_ts: datetime) -> pd.DataFrame:
        """All visits to a station within [start_ts, end_ts)."""
        e = self.events
        mask = (e.station_id == station_id) & (e.entry_ts >= start_ts) & (e.entry_ts < end_ts)
        return e[mask]

    def is_downstream(self, station_id: str, of_station_id: str) -> bool:
        return self._order_index.get(station_id, -1) > self._order_index.get(of_station_id, -1)

    def vins_in_transit_to(self, station_id: str, target_station_id: str,
                            start_ts: datetime, end_ts: datetime) -> pd.DataFrame:
        """VINs that passed `station_id` in the window and have NOT yet reached
        `target_station_id` (the downstream inspection point). Returns one row
        per such VIN, with its last known station/ts."""
        window = self.station_window(station_id, start_ts, end_ts)
        if window.empty:
            return window.assign(last_station=[], last_ts=[])

        rows = []
        for _, visit in window.iterrows():
            vin = visit["vin"]
            if self.has_reached(vin, target_station_id):
                continue
            last = self.last_known_station(vin)
            rows.append({
                "vin": vin,
                "variant": visit["variant"],
                "origin_station": station_id,
                "origin_entry_ts": visit["entry_ts"],
                "last_station": last[0] if last else station_id,
                "last_ts": last[1] if last else visit["exit_ts"],
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Transit time estimation (for ETA calculations in infer.py)
    # ------------------------------------------------------------------

    def median_transit_seconds(self, from_station: str, to_station: str) -> float:
        """Empirical median time from exiting `from_station` to entering
        `to_station`, across all VINs that visited both. Cached."""
        key = (from_station, to_station)
        if key in self._transit_cache:
            return self._transit_cache[key]

        from_visits = self.events[self.events.station_id == from_station][["vin", "exit_ts"]]
        to_visits = self.events[self.events.station_id == to_station][["vin", "entry_ts"]]
        merged = from_visits.merge(to_visits, on="vin", how="inner")
        if merged.empty:
            # Fall back to a rough estimate: nominal cycle time * number of intervening stations.
            i_from = self._order_index.get(from_station)
            i_to = self._order_index.get(to_station)
            n_stations = max(1, (i_to - i_from)) if i_from is not None and i_to is not None else 1
            estimate = n_stations * 60.0
            self._transit_cache[key] = estimate
            return estimate

        deltas = (merged["entry_ts"] - merged["exit_ts"]).dt.total_seconds()
        deltas = deltas[deltas >= 0]
        estimate = float(deltas.median()) if len(deltas) else 60.0
        self._transit_cache[key] = estimate
        return estimate

    def eta_to_station(self, vin: str, target_station_id: str) -> Optional[pd.Timestamp]:
        last = self.last_known_station(vin)
        if last is None:
            return None
        last_station, last_ts = last
        if self._order_index.get(last_station, -1) >= self._order_index.get(target_station_id, 10**9):
            return last_ts  # already at or past target
        transit_s = self.median_transit_seconds(last_station, target_station_id)
        return last_ts + timedelta(seconds=transit_s)
