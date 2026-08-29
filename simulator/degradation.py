"""Degradation (drift) processes layered onto station signals.

Implements the SignalProvider hook from simulator/line.py. Produces a mix of
episode types so the forecaster (models/forecast.py) can later be honestly
tested against ground truth: linear_wear, accelerating_wear, step_change,
noise_only, recovering. See BUILD_SPEC.md §5.3.

Ground truth for each episode (used only by validation, never by models) is
exposed via DegradationEngine.truth_records() -> degradation_truth.parquet.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from simulator.line import SignalProvider, StationConfig

EPISODE_TYPES = ("linear_wear", "accelerating_wear", "step_change", "noise_only", "recovering")

DEFAULT_TYPE_WEIGHTS = {
    "linear_wear": 0.35,
    "accelerating_wear": 0.15,
    "step_change": 0.20,
    "noise_only": 0.20,
    "recovering": 0.10,
}


@dataclass
class DegradationEpisode:
    episode_id: str
    station_id: str
    signal: str
    episode_type: str
    start_ts: datetime
    nominal: float
    noise_sd: float
    spec_low: float
    spec_high: float
    drift_rate_per_hour: float = 0.0
    accel_per_hour2: float = 0.0
    step_offset: float = 0.0
    maintenance_ts: Optional[datetime] = None

    def _target_limit(self) -> float:
        rate = self.step_offset if self.episode_type == "step_change" else self.drift_rate_per_hour
        return self.spec_high if rate >= 0 else self.spec_low

    def mean_value_at(self, ts: datetime) -> float:
        if self.maintenance_ts is not None and ts >= self.maintenance_ts:
            return self.nominal
        elapsed_h = max(0.0, (ts - self.start_ts).total_seconds() / 3600.0)
        if self.episode_type in ("linear_wear", "recovering"):
            return self.nominal + self.drift_rate_per_hour * elapsed_h
        if self.episode_type == "accelerating_wear":
            return self.nominal + self.drift_rate_per_hour * elapsed_h + self.accel_per_hour2 * elapsed_h ** 2
        if self.episode_type == "step_change":
            return self.nominal + self.step_offset
        return self.nominal  # noise_only

    def actual_spec_crossing_ts(self) -> Optional[datetime]:
        """Ground truth crossing time, ignoring noise. None if it never crosses
        (noise_only always None; others None if maintenance intervenes first)."""
        limit = self._target_limit()

        if self.episode_type == "noise_only":
            return None

        if self.episode_type == "step_change":
            val = self.nominal + self.step_offset
            crossed = val >= limit if self.step_offset >= 0 else val <= limit
            return self.start_ts if crossed else None

        if self.episode_type in ("linear_wear", "recovering"):
            if self.drift_rate_per_hour == 0:
                return None
            hours = (limit - self.nominal) / self.drift_rate_per_hour
            if hours < 0:
                return None
            crossing = self.start_ts + timedelta(hours=hours)
            if self.maintenance_ts is not None and crossing >= self.maintenance_ts:
                return None
            return crossing

        if self.episode_type == "accelerating_wear":
            a, b = self.drift_rate_per_hour, self.accel_per_hour2
            c = self.nominal - limit
            if b == 0:
                if a == 0:
                    return None
                hours = -c / a
                positive = [hours] if hours > 0 else []
            else:
                disc = a * a - 4 * b * c
                if disc < 0:
                    return None
                sq = math.sqrt(disc)
                roots = [(-a + sq) / (2 * b), (-a - sq) / (2 * b)]
                positive = [r for r in roots if r > 0]
            if not positive:
                return None
            hours = min(positive)
            crossing = self.start_ts + timedelta(hours=hours)
            if self.maintenance_ts is not None and crossing >= self.maintenance_ts:
                return None
            return crossing

        return None


class DegradationEngine(SignalProvider):
    """SignalProvider that injects scheduled drift episodes on top of nominal+noise."""

    def __init__(self, station_configs: list[StationConfig]):
        self.station_configs = station_configs
        self._signal_specs = {
            (s.id, sig_name): spec
            for s in station_configs
            for sig_name, spec in (s.signals or {}).items()
        }
        self.zone_of = {s.id: s.zone for s in station_configs}
        self.episodes: list[DegradationEpisode] = []
        self._counter = 0

    def instrumented_signal_pairs(self) -> list[tuple[str, str]]:
        return list(self._signal_specs.keys())

    def schedule_episode(self, station_id: str, signal: str, episode_type: str,
                          start_ts: datetime, drift_rate_per_hour: float = 0.0,
                          accel_per_hour2: float = 0.0, step_offset: float = 0.0,
                          maintenance_ts: Optional[datetime] = None) -> DegradationEpisode:
        if episode_type not in EPISODE_TYPES:
            raise ValueError(f"unknown episode_type {episode_type!r}")
        spec = self._signal_specs[(station_id, signal)]
        episode_id = f"EP{self._counter:05d}"
        self._counter += 1
        ep = DegradationEpisode(
            episode_id=episode_id, station_id=station_id, signal=signal,
            episode_type=episode_type, start_ts=start_ts,
            nominal=spec["nominal"], noise_sd=spec["noise_sd"],
            spec_low=spec["spec_low"], spec_high=spec["spec_high"],
            drift_rate_per_hour=drift_rate_per_hour, accel_per_hour2=accel_per_hour2,
            step_offset=step_offset, maintenance_ts=maintenance_ts,
        )
        self.episodes.append(ep)
        return ep

    def generate_background_episodes(self, start_ts: datetime, end_ts: datetime,
                                      rng: random.Random,
                                      episodes_per_signal_per_day: float = 0.15,
                                      type_weights: Optional[dict] = None) -> None:
        """Poisson-ish scatter of episodes across all instrumented (station, signal)
        pairs, for 'nominal'/'noisy_line' style multi-day scenarios."""
        type_weights = type_weights or DEFAULT_TYPE_WEIGHTS
        types = list(type_weights.keys())
        weights = list(type_weights.values())
        days = max(0.0, (end_ts - start_ts).total_seconds() / 86400.0)

        for (station_id, signal) in self._signal_specs.keys():
            spec = self._signal_specs[(station_id, signal)]
            span = spec["spec_high"] - spec["spec_low"]
            expected_n = episodes_per_signal_per_day * days
            n = rng.gauss(expected_n, max(0.5, expected_n * 0.4))
            n = max(0, round(n))
            # Space episodes out across the horizon so they don't overlap for one signal.
            if n == 0:
                continue
            slot_h = (days * 24) / n
            cursor = start_ts
            for _ in range(n):
                jitter_h = rng.uniform(0, max(1.0, slot_h * 0.6))
                ep_start = cursor + timedelta(hours=jitter_h)
                if ep_start >= end_ts:
                    break
                etype = rng.choices(types, weights=weights, k=1)[0]
                self._add_random_episode(station_id, signal, etype, ep_start, span, rng)
                cursor += timedelta(hours=slot_h)

    def _add_random_episode(self, station_id, signal, etype, start_ts, span, rng: random.Random):
        # magnitudes scaled to the signal's spec span so they're meaningful regardless
        # of the signal's absolute units
        if etype == "linear_wear":
            rate = rng.choice([-1, 1]) * span * rng.uniform(0.03, 0.12)  # per hour
            self.schedule_episode(station_id, signal, etype, start_ts, drift_rate_per_hour=rate)
        elif etype == "accelerating_wear":
            rate = rng.choice([-1, 1]) * span * rng.uniform(0.01, 0.04)
            accel = math.copysign(span * rng.uniform(0.003, 0.012), rate)
            self.schedule_episode(station_id, signal, etype, start_ts,
                                   drift_rate_per_hour=rate, accel_per_hour2=accel)
        elif etype == "step_change":
            offset = rng.choice([-1, 1]) * span * rng.uniform(0.15, 0.4)
            self.schedule_episode(station_id, signal, etype, start_ts, step_offset=offset)
        elif etype == "recovering":
            rate = rng.choice([-1, 1]) * span * rng.uniform(0.05, 0.15)
            maint_h = rng.uniform(1.0, 5.0)
            self.schedule_episode(station_id, signal, etype, start_ts, drift_rate_per_hour=rate,
                                   maintenance_ts=start_ts + timedelta(hours=maint_h))
        else:  # noise_only: a "quiet episode" marker so validation has explicit windows
            self.schedule_episode(station_id, signal, etype, start_ts)

    def _active_episode(self, station_id: str, signal: str, ts: datetime) -> Optional[DegradationEpisode]:
        candidates = [
            e for e in self.episodes
            if e.station_id == station_id and e.signal == signal and e.start_ts <= ts
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda e: e.start_ts)

    def mean_value_at_ts(self, station_id: str, signal: str, ts: datetime) -> float:
        ep = self._active_episode(station_id, signal, ts)
        if ep is None:
            return self._signal_specs[(station_id, signal)]["nominal"]
        return ep.mean_value_at(ts)

    # --- SignalProvider interface ---
    def sample(self, station_id: str, signal_name: str, nominal: float, noise_sd: float,
               ts: datetime, rng: random.Random) -> float:
        ep = self._active_episode(station_id, signal_name, ts)
        if ep is None:
            return rng.gauss(nominal, noise_sd)
        return rng.gauss(ep.mean_value_at(ts), ep.noise_sd)

    def truth_records(self) -> list[dict]:
        return [
            {
                "episode_id": e.episode_id,
                "station_id": e.station_id,
                "signal": e.signal,
                "start_ts": e.start_ts,
                "drift_rate_per_hour": e.drift_rate_per_hour,
                "actual_spec_crossing_ts": e.actual_spec_crossing_ts(),
                "maintenance_ts": e.maintenance_ts,
                "episode_type": e.episode_type,
            }
            for e in self.episodes
        ]
