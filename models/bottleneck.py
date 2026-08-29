"""Bottleneck detection via Roser's active-period method (BUILD_SPEC.md §7.2).

The momentary bottleneck is whichever currently-working station has the longest
uninterrupted active streak. Operates on the run-length-encoded status log from
simulator/line.py (station_id, status, start_ts, end_ts).
"""
from __future__ import annotations

from datetime import timedelta

import pandas as pd


def bottleneck_migration(status_df: pd.DataFrame) -> pd.DataFrame:
    """Returns contiguous (start_ts, end_ts, station_id) bottleneck assignments.
    O(n log n) sweep: track each station's ongoing 'working' streak start time as
    we cross interval boundaries; the momentary bottleneck is whoever's streak
    (now - streak_start) is longest among stations currently working."""
    working = status_df[status_df.status == "working"]
    if working.empty:
        return pd.DataFrame(columns=["start_ts", "end_ts", "station_id"])

    events = []
    for row in working.itertuples():
        events.append((row.start_ts, 1, row.station_id, row.start_ts))
        events.append((row.end_ts, 0, row.station_id, row.start_ts))
    events.sort(key=lambda e: (e[0], e[1]))  # ends (0) before starts (1) at equal ts

    active: dict[str, pd.Timestamp] = {}
    segments = []
    cur_station, cur_start, prev_t = None, None, None
    for t, typ, station, streak_start in events:
        if prev_t is not None and t > prev_t and active:
            bottleneck_now = max(active, key=lambda s: prev_t - active[s])
            if bottleneck_now != cur_station:
                if cur_station is not None:
                    segments.append({"start_ts": cur_start, "end_ts": prev_t, "station_id": cur_station})
                cur_station, cur_start = bottleneck_now, prev_t
        if typ == 1:
            active[station] = streak_start
        else:
            active.pop(station, None)
        prev_t = t
    if cur_station is not None and prev_t is not None:
        segments.append({"start_ts": cur_start, "end_ts": prev_t, "station_id": cur_station})
    return pd.DataFrame(segments)


def bottleneck_hours_by_station(migration_df: pd.DataFrame) -> pd.Series:
    """Total bottleneck-time per station, for the manager heatmap's station axis."""
    if migration_df.empty:
        return pd.Series(dtype=float)
    dur_h = (migration_df.end_ts - migration_df.start_ts).dt.total_seconds() / 3600.0
    return migration_df.assign(dur_h=dur_h).groupby("station_id").dur_h.sum().sort_values(ascending=False)


def buffer_trend_projection(status_df: pd.DataFrame, station_id: str, now_ts,
                             lookback_min: int = 30, block_frac_threshold: float = 0.5):
    """Rough short-horizon projection: minutes until `station_id` is chronically
    blocked (a practical proxy for "breaches takt", per BUILD_SPEC.md §7.2).

    We don't log literal buffer occupancy — only station status intervals (see
    ASSUMPTIONS.md) — so blocked-time fraction over a trailing window is used as
    a proxy: a station blocks more often as its downstream buffer fills. Compares
    two successive lookback windows and linearly extrapolates the trend.
    Returns minutes-to-threshold, 0.0 if already over threshold, or None if not
    trending toward it / insufficient data.
    """
    span_start = now_ts - timedelta(minutes=2 * lookback_min)
    window = status_df[
        (status_df.station_id == station_id) & (status_df.end_ts > span_start) & (status_df.start_ts < now_ts)
    ]
    if window.empty:
        return None

    mid = now_ts - timedelta(minutes=lookback_min)

    def blocked_frac(a, b):
        seg = window[(window.end_ts > a) & (window.start_ts < b) & (window.status == "blocked")]
        total_s = (b - a).total_seconds()
        if total_s <= 0:
            return 0.0
        blocked_s = sum(
            (min(r.end_ts, b) - max(r.start_ts, a)).total_seconds() for r in seg.itertuples()
        )
        return blocked_s / total_s

    f1 = blocked_frac(span_start, mid)
    f2 = blocked_frac(mid, now_ts)
    if f2 >= block_frac_threshold:
        return 0.0
    rate_per_min = (f2 - f1) / lookback_min
    if rate_per_min <= 0:
        return None
    return (block_frac_threshold - f2) / rate_per_min
