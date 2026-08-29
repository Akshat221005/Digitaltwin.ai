"""Cascade simulation (BUILD_SPEC.md §7.4): projects the downstream consequences
of leaving a forecasted degradation unaddressed, and compares against a proposed
intervention (the counterfactual that justifies the recommendation).

Reuses simulator/line.py's own mechanics by forking the live LineSimulator's
runtime state and continuing it forward — no second mechanics engine, per the
build spec's explicit instruction.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from simulator.line import LineSimulator


@dataclass
class CascadeResult:
    label: str
    failure_ts: datetime
    down_duration_s: float
    time_to_first_starvation: Optional[datetime]
    stoppage_duration_s: float
    units_completed: int
    rework_cost: float
    status_df: pd.DataFrame
    events_df: pd.DataFrame


def _fork(sim: LineSimulator) -> LineSimulator:
    """Deep-copies the simulator's full runtime state. The twin never mutates the
    live line — only a fork of it, which is discarded after the projection."""
    return copy.deepcopy(sim)


def run_single_cascade(sim: LineSimulator, station_id: str, failure_ts: datetime,
                        down_duration_s: float, horizon_s: int, label: str,
                        rework_cost_per_unit: float = 150.0,
                        reset_signal: Optional[tuple] = None) -> CascadeResult:
    """Forks `sim`, takes `station_id` down for `down_duration_s` starting at
    `failure_ts`, runs the fork forward `horizon_s` more seconds, and measures
    the consequences: first downstream starvation, stoppage duration, units
    still completed, and rework cost from defects surfacing in the window.

    `reset_signal`, if given, is a (station_id, signal) pair whose currently
    active degradation episode gets its maintenance_ts set to `failure_ts` on
    the fork — i.e. this cascade represents an actual repair (electrode
    replaced, nozzle cleared), not just a scheduling gap. Without this, taking
    a station mechanically "down" has no effect on the signal that's actually
    driving defects, which would make an intervention look pointless.
    """
    fork = _fork(sim)
    if fork.current_ts is None:
        raise ValueError("sim must have been run() at least once before forking")

    order = [s.cfg.id for s in fork.stations]
    idx = order.index(station_id)
    downstream_ids = order[idx + 1:]
    last_station_id = order[-1]

    fork.by_id[station_id].down_until = failure_ts + timedelta(seconds=down_duration_s)

    if reset_signal is not None and hasattr(fork.signal_provider, "_active_episode"):
        r_station, r_signal = reset_signal
        ep = fork.signal_provider._active_episode(r_station, r_signal, failure_ts)
        if ep is not None:
            ep.maintenance_ts = failure_ts

    resume_ts = fork.current_ts
    fork.run(start_ts=resume_ts, duration_s=horizon_s)

    status_df = fork.status_df()
    window = status_df[status_df.start_ts >= resume_ts]

    starvation = window[
        window.station_id.isin(downstream_ids) & (window.status == "starved")
    ].sort_values("start_ts")
    time_to_first_starvation = starvation.iloc[0]["start_ts"] if len(starvation) else None

    down_intervals = window[(window.station_id == station_id) & (window.status == "down")]
    stoppage_duration_s = float((down_intervals.end_ts - down_intervals.start_ts).dt.total_seconds().sum())

    events_df = fork.events_df()
    units_completed = int((events_df.station_id == last_station_id).sum())

    defects_df = fork.defects_df()
    n_defects_in_window = 0
    if not defects_df.empty:
        n_defects_in_window = int(
            (defects_df["injected_ts"] >= failure_ts).sum()
        )
    rework_cost = n_defects_in_window * rework_cost_per_unit

    return CascadeResult(
        label=label, failure_ts=failure_ts, down_duration_s=down_duration_s,
        time_to_first_starvation=time_to_first_starvation,
        stoppage_duration_s=stoppage_duration_s, units_completed=units_completed,
        rework_cost=rework_cost, status_df=status_df, events_df=events_df,
    )


def compare_with_and_without_intervention(
    sim: LineSimulator, station_id: str, signal: str,
    unplanned_failure_ts: datetime, unplanned_down_duration_s: float,
    planned_maintenance_ts: datetime, planned_down_duration_s: float,
    horizon_s: int, rework_cost_per_unit: float = 150.0,
) -> dict:
    """The counterfactual pair BUILD_SPEC.md §7.4 calls for: what happens if the
    forecasted drift is ignored (unplanned failure once it crosses spec) versus
    what happens if the recommended maintenance is scheduled proactively (a
    short planned window before the crossing, which also fixes `signal`'s
    underlying drift). units_protected is the throughput difference between the
    two; rework_cost_avoided is the defect-cost difference — together, the
    numbers that justify the recommendation."""
    do_nothing = run_single_cascade(
        sim, station_id, unplanned_failure_ts, unplanned_down_duration_s, horizon_s,
        label="do_nothing", rework_cost_per_unit=rework_cost_per_unit,
    )
    with_intervention = run_single_cascade(
        sim, station_id, planned_maintenance_ts, planned_down_duration_s, horizon_s,
        label="with_intervention", rework_cost_per_unit=rework_cost_per_unit,
        reset_signal=(station_id, signal),
    )
    units_protected = max(0, with_intervention.units_completed - do_nothing.units_completed)
    rework_avoided = max(0.0, do_nothing.rework_cost - with_intervention.rework_cost)
    return {
        "do_nothing": do_nothing,
        "with_intervention": with_intervention,
        "units_protected": units_protected,
        "rework_cost_avoided": rework_avoided,
    }
