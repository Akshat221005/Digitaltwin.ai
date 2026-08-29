"""Time-stepped mixed-model assembly line simulator.

Ticks at 1-second resolution. Stations are working / blocked / starved / down.
Standard buffer transfer between stations in a fixed sequence (by station id).

This module owns line *mechanics* only. Degradation (drift) and latent defects
are layered on top by simulator/degradation.py and simulator/defects.py via the
SignalProvider and DefectInjector hooks below, so this file has no knowledge of
either — it just asks the hooks for a value/decision at the right moment.
"""
from __future__ import annotations

import hashlib
import random
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Hooks other modules plug into. Defaults here make line.py fully runnable
# on its own (Build order step 1 must work before degradation/defects exist).
# ---------------------------------------------------------------------------

class SignalProvider:
    """Produces a signal reading for (station_id, signal_name) at a given ts.

    Default implementation: nominal + gaussian noise, i.e. a perfectly healthy
    station. degradation.py subclasses/replaces this to inject drift.
    """

    def sample(self, station_id: str, signal_name: str, nominal: float, noise_sd: float,
               ts: datetime, rng: random.Random) -> float:
        return rng.gauss(nominal, noise_sd)


class DefectInjector:
    """Decides whether a vehicle passing a station right now gets a latent defect.

    Default implementation never injects anything. defects.py replaces this.
    Returns (defect_type, root_cause) or None.
    """

    def maybe_injure(self, station_id: str, vin: str, variant: str, ts: datetime,
                      rng: random.Random) -> Optional[dict]:
        return None


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

@dataclass
class StationConfig:
    id: str
    name: str
    zone: str
    sensor_tier: str
    base_cycle_time_s: float
    cycle_time_cv: float
    buffer_before: int
    variant_cycle_multiplier: dict
    signals: dict = field(default_factory=dict)


@dataclass
class LineConfig:
    zones: list
    variants: list
    inspection_stations: dict
    stations: list  # list[StationConfig], in line order


def load_line_config(path: str = "config/line.yaml") -> LineConfig:
    with open(path) as f:
        doc = yaml.safe_load(f)
    stations = [StationConfig(**s) for s in doc["stations"]]
    stations.sort(key=lambda s: int(s.id[1:]))
    return LineConfig(
        zones=doc["line"]["zones"],
        variants=doc["line"]["variants"],
        inspection_stations=doc["line"]["inspection_stations"],
        stations=stations,
    )


# ---------------------------------------------------------------------------
# Runtime station / vehicle state
# ---------------------------------------------------------------------------

STATUS_WORKING = "working"
STATUS_BLOCKED = "blocked"
STATUS_STARVED = "starved"
STATUS_DOWN = "down"
STATUS_IDLE = "idle"  # transient: buffer empty check not yet resolved this tick


@dataclass
class Vehicle:
    vin: str
    variant: str
    launched_ts: datetime


@dataclass
class InProcess:
    vehicle: Vehicle
    entry_ts: datetime
    finish_ts: datetime
    operator_id: str


@dataclass
class StationRuntime:
    cfg: StationConfig
    buffer: deque = field(default_factory=deque)      # vehicles waiting to enter
    processing: Optional[InProcess] = None
    finished_waiting: Optional[InProcess] = None       # done, waiting for downstream space
    status: str = STATUS_STARVED
    cycle_count: int = 0                               # for tier-B 1-in-5 sampling
    down_until: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

SHIFT_LENGTH_H = 8
N_OPERATORS_PER_STATION = 2  # rotate across shifts


def shift_id_for(ts: datetime) -> str:
    hour_of_day = ts.hour + ts.minute / 60
    shift_num = int(hour_of_day // SHIFT_LENGTH_H)  # 0, 1, 2
    return f"{ts.date().isoformat()}-shift{shift_num}"


def operator_id_for(station_id: str, ts: datetime, rng: random.Random) -> str:
    shift_num = int((ts.hour + ts.minute / 60) // SHIFT_LENGTH_H)
    op_index = shift_num % N_OPERATORS_PER_STATION
    return f"OP-{station_id}-{op_index}"


def _station_seed(seed: int, station_id: str) -> int:
    """Deterministic per-station seed. NOT `hash((seed, station_id))` — Python
    randomizes str hashing per-process (PYTHONHASHSEED) unless disabled, which
    would make this look deterministic within one run but differ across runs,
    silently breaking the "byte-reproducible" guarantee scenarios depend on.
    sha256 is stable across processes and interpreter versions."""
    digest = hashlib.sha256(f"{seed}:{station_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


class LineSimulator:
    def __init__(self, cfg: LineConfig, seed: int = 42,
                 signal_provider: Optional[SignalProvider] = None,
                 defect_injector: Optional[DefectInjector] = None,
                 launch_interval_s: Optional[float] = None,
                 variant_sequence: Optional[Callable[[int], str]] = None):
        self.cfg = cfg
        # One independent RNG stream per station, not one shared stream for the
        # whole line. With a single shared stream, any perturbation to one
        # station (e.g. going down for a cascade projection) shifts every later
        # draw for every OTHER station too — since skipping that station's draw
        # for one tick shifts everyone else's position in the same sequence —
        # making two forked runs diverge everywhere almost immediately, for
        # reasons unrelated to real downstream causality. That silently broke
        # models/cascade.py's do-nothing-vs-intervention comparison: "affected
        # stations" came back as nearly all 41, dominated by RNG artifact, not
        # the real, much smaller set actually downstream of a failure. A
        # perturbation to one station's stream now can only affect others
        # through the real buffer/blocking mechanics, which is what a
        # counterfactual comparison needs.
        self.station_rngs = {s.id: random.Random(_station_seed(seed, s.id)) for s in cfg.stations}
        self.signal_provider = signal_provider or SignalProvider()
        self.defect_injector = defect_injector or DefectInjector()
        self.stations = [StationRuntime(cfg=s) for s in cfg.stations]
        self.by_id = {s.cfg.id: s for s in self.stations}
        # Launch cadence defaults to the slowest nominal cycle time across stations,
        # i.e. line takt, so the line runs near-saturated without being force-fed.
        self.launch_interval_s = launch_interval_s or max(
            s.base_cycle_time_s for s in cfg.stations
        )
        self.variant_sequence = variant_sequence or self._default_variant_sequence
        self._next_vin = 1
        self._time_since_last_launch = 10 ** 9  # force immediate first launch

        self.events: list[dict] = []
        self.ground_truth_defects: list[dict] = []
        self.status_intervals: list[dict] = []  # run-length encoded station status log
        self._open_interval: dict[str, dict] = {}  # station_id -> {"status", "start_ts"}
        self.current_ts: Optional[datetime] = None  # lets callers (e.g. cascade forks) resume run()

    def _default_variant_sequence(self, n: int) -> str:
        # Repeating pattern with sedan-majority, matches BUILD_SPEC's variant-mix story.
        pattern = ["sedan", "sedan", "suv", "sedan", "van", "suv", "sedan"]
        return pattern[n % len(pattern)]

    def _new_vin(self) -> str:
        vin = f"VIN{self._next_vin:07d}"
        self._next_vin += 1
        return vin

    def _record_status(self, station_id: str, status: str, ts: datetime):
        cur = self._open_interval.get(station_id)
        if cur is not None and cur["status"] == status:
            return  # unchanged, extend implicitly (we backfill end_ts on close)
        if cur is not None:
            self.status_intervals.append({
                "station_id": station_id,
                "status": cur["status"],
                "start_ts": cur["start_ts"],
                "end_ts": ts,
            })
        self._open_interval[station_id] = {"status": status, "start_ts": ts}

    def _close_all_intervals(self, ts: datetime):
        # Closes each open interval for reporting, then re-opens it at `ts` so a
        # subsequent run() call (e.g. a cascade fork resuming from here) doesn't
        # produce a stale/overlapping interval on its first status change.
        for station_id, cur in self._open_interval.items():
            self.status_intervals.append({
                "station_id": station_id,
                "status": cur["status"],
                "start_ts": cur["start_ts"],
                "end_ts": ts,
            })
            cur["start_ts"] = ts

    def _cycle_time(self, sr: StationRuntime, variant: str, ts: datetime) -> float:
        base = sr.cfg.base_cycle_time_s * sr.cfg.variant_cycle_multiplier.get(variant, 1.0)
        noisy = max(1.0, self.station_rngs[sr.cfg.id].gauss(base, base * sr.cfg.cycle_time_cv))
        # Fatigue: dwell inflates late in an 8h shift (peaks near end of shift).
        minutes_into_shift = ((ts.hour * 60 + ts.minute) % (SHIFT_LENGTH_H * 60))
        fatigue_frac = minutes_into_shift / (SHIFT_LENGTH_H * 60)
        fatigue_mult = 1.0 + 0.05 * max(0.0, fatigue_frac - 0.6)  # up to +5% late shift
        return noisy * fatigue_mult

    def _sample_signals(self, sr: StationRuntime, ts: datetime) -> dict:
        out = {}
        if not sr.cfg.signals:
            return out
        tier = sr.cfg.sensor_tier
        if tier == "C":
            return out
        if tier == "B" and (sr.cycle_count % 5 != 0):
            return out  # tier B sampled every 5th cycle only
        for sig_name, spec in sr.cfg.signals.items():
            val = self.signal_provider.sample(
                sr.cfg.id, sig_name, spec["nominal"], spec["noise_sd"], ts, self.station_rngs[sr.cfg.id]
            )
            out[sig_name] = val
        return out

    def _try_start_processing(self, sr: StationRuntime, ts: datetime):
        if sr.processing is not None or sr.finished_waiting is not None:
            return
        if sr.down_until is not None and ts < sr.down_until:
            sr.status = STATUS_DOWN
            return
        if not sr.buffer:
            sr.status = STATUS_STARVED
            return
        vehicle = sr.buffer.popleft()
        cycle_s = self._cycle_time(sr, vehicle.variant, ts)
        operator = operator_id_for(sr.cfg.id, ts, self.station_rngs[sr.cfg.id])
        sr.processing = InProcess(
            vehicle=vehicle, entry_ts=ts,
            finish_ts=ts + timedelta(seconds=cycle_s),
            operator_id=operator,
        )
        sr.status = STATUS_WORKING

    def _finish_processing_if_due(self, sr: StationRuntime, ts: datetime):
        if sr.processing is not None and ts >= sr.processing.finish_ts:
            sr.finished_waiting = sr.processing
            sr.processing = None
            sr.cycle_count += 1

    def _try_advance_to_next(self, sr: StationRuntime, next_sr: Optional[StationRuntime], ts: datetime):
        if sr.finished_waiting is None:
            return
        ip = sr.finished_waiting
        can_move = (next_sr is None) or (len(next_sr.buffer) < next_sr.cfg.buffer_before)
        if not can_move:
            sr.status = STATUS_BLOCKED
            return
        # record the completed visit
        signals = self._sample_signals(sr, ip.entry_ts)
        row = {
            "vin": ip.vehicle.vin,
            "station_id": sr.cfg.id,
            "variant": ip.vehicle.variant,
            "entry_ts": ip.entry_ts,
            "exit_ts": ts,
            "dwell_s": (ts - ip.entry_ts).total_seconds(),
            "operator_id": ip.operator_id,
            "shift_id": shift_id_for(ip.entry_ts),
        }
        row.update(signals)
        self.events.append(row)

        defect = self.defect_injector.maybe_injure(
            sr.cfg.id, ip.vehicle.vin, ip.vehicle.variant, ip.entry_ts, self.station_rngs[sr.cfg.id]
        )
        if defect is not None:
            defect["vin"] = ip.vehicle.vin
            defect["injected_at_station"] = sr.cfg.id
            defect["injected_ts"] = ip.entry_ts
            self.ground_truth_defects.append(defect)

        if next_sr is not None:
            next_sr.buffer.append(ip.vehicle)
        sr.finished_waiting = None

    def _launch_if_due(self, ts: datetime, dt_s: float):
        first = self.stations[0]
        self._time_since_last_launch += dt_s
        if self._time_since_last_launch < self.launch_interval_s:
            return
        if len(first.buffer) >= first.cfg.buffer_before:
            return  # line pull is full, hold the launch (models upstream stamping/marshalling)
        n = self._next_vin - 1
        variant = self.variant_sequence(n)
        vin = self._new_vin()
        first.buffer.append(Vehicle(vin=vin, variant=variant, launched_ts=ts))
        self._time_since_last_launch = 0.0

    def run(self, start_ts: datetime, duration_s: int, tick_s: int = 1) -> dict:
        ts = start_ts
        for _ in range(0, duration_s, tick_s):
            self._launch_if_due(ts, tick_s)

            for sr in self.stations:
                self._finish_processing_if_due(sr, ts)

            for i, sr in enumerate(self.stations):
                next_sr = self.stations[i + 1] if i + 1 < len(self.stations) else None
                self._try_advance_to_next(sr, next_sr, ts)

            for sr in self.stations:
                self._try_start_processing(sr, ts)
                self._record_status(sr.cfg.id, sr.status, ts)

            ts += timedelta(seconds=tick_s)

        self._close_all_intervals(ts)
        self.current_ts = ts
        return {"end_ts": ts}

    def events_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.events)

    def status_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.status_intervals)

    def defects_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.ground_truth_defects)
