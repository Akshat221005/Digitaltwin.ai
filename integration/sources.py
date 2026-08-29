"""Integration layer (BUILD_SPEC.md §9): minimal, and mostly documentation.

There is no writer interface anywhere in this codebase. That absence is the
point — the twin is read-only by construction, not just by convention.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterator, Protocol

import pandas as pd


@dataclass
class Event:
    vin: str
    station_id: str
    variant: str
    entry_ts: datetime
    exit_ts: datetime
    dwell_s: float
    operator_id: str
    shift_id: str
    signals: dict  # empty for tier-C stations, by design (simulator/line.py)


class DataSource(Protocol):
    def stream(self) -> Iterator[Event]: ...
    def health(self) -> dict: ...


class SimulatedSource:
    """Replays a pre-generated scenario's events.parquet as an Event stream, in
    entry_ts order. This is what everything else in this repo actually runs
    against — the "real, used by everything" implementation per BUILD_SPEC.md §9."""

    def __init__(self, events_df: pd.DataFrame, signal_columns: list[str]):
        self.events_df = events_df.sort_values("entry_ts")
        self.signal_columns = signal_columns
        self._n_emitted = 0

    def stream(self) -> Iterator[Event]:
        for row in self.events_df.itertuples():
            signals = {
                col: getattr(row, col) for col in self.signal_columns
                if hasattr(row, col) and pd.notna(getattr(row, col))
            }
            self._n_emitted += 1
            yield Event(
                vin=row.vin, station_id=row.station_id, variant=row.variant,
                entry_ts=row.entry_ts, exit_ts=row.exit_ts, dwell_s=row.dwell_s,
                operator_id=row.operator_id, shift_id=row.shift_id, signals=signals,
            )

    def health(self) -> dict:
        return {
            "source": "simulated", "status": "ok",
            "events_emitted": self._n_emitted, "total_events": len(self.events_df),
        }


class OPCUASource:
    """Stub for a real read-only OPC-UA tap. Deliberately not implemented —
    BUILD_SPEC.md §3 rules out a real OPC-UA client for this prototype's scope.
    Documents the interface a real deployment would need:

    - Tag mapping: one OPC-UA node per (station_id, signal) for tier-A/B stations
      (e.g. ns=2;s=Line1.S12.WeldCurrentA), plus a station entry/exit barcode-scan
      subscription for every station, tier C included — barcode scans are the
      ONLY signal tier-C stations provide, by design (BUILD_SPEC.md §5.1).
    - Polling: tier-A signals via a subscription (report-by-exception, ~1s
      publishing interval, matching the simulator's own tick rate); tier-B
      signals polled every 5th cycle, matching the simulator's sampling
      assumption; barcode events via subscription only (discrete, not periodic).
    - Placement: an edge gateway inside the plant network reads the OPC-UA
      server (or an MQTT bridge, where a historian already publishes one) and
      forwards to the twin over a one-way connection. No path back into line
      control exists anywhere in this codebase — there is no writer interface
      to implement even if this stub were filled in.
    """

    def stream(self) -> Iterator[Event]:
        raise NotImplementedError(
            "OPC-UA integration is out of scope for this prototype (BUILD_SPEC.md §3). "
            "See this class's docstring for the tag mapping, polling interval, and "
            "edge-gateway placement a real integration would use."
        )

    def health(self) -> dict:
        raise NotImplementedError("see stream()")
