"""CLI: generate simulated assembly-line data for a named scenario
(BUILD_SPEC.md §5.5). Writes events.parquet, ground_truth.parquet,
degradation_truth.parquet, and status_intervals.parquet to data/<scenario>/.

Usage:
    python -m simulator.run --scenario weld_drift_demo
    python -m simulator.run --scenario nominal --out data/nominal
"""
from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml

from simulator.defects import LatentDefectInjector, surface_defects
from simulator.degradation import DegradationEngine
from simulator.line import LineSimulator, load_line_config


def load_scenarios(path: str = "config/scenarios.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)["scenarios"]


def _scenario_duration_s(scenario: dict) -> int:
    if "duration_days" in scenario:
        return int(scenario["duration_days"] * 86400)
    return int(scenario["duration_hours"] * 3600)


def build_scenario(cfg, scenario: dict, run_duration_s: int = None):
    """Builds and runs a scenario. `run_duration_s`, if given, runs the sim for
    fewer ticks than the scenario's full horizon (e.g. to fork a live state at
    a specific point in time for cascade projection) — but degradation episodes
    are still scheduled against the FULL scenario horizon regardless, so a
    partial run produces byte-identical signal behaviour to the full run up to
    that point, just truncated."""
    engine = DegradationEngine(cfg.stations)
    start_ts = datetime.fromisoformat(scenario["start_ts"])
    full_duration_s = _scenario_duration_s(scenario)
    end_ts = start_ts + timedelta(seconds=full_duration_s)
    duration_s = full_duration_s if run_duration_s is None else min(run_duration_s, full_duration_s)

    bg_rate = scenario.get("background_episodes_per_signal_per_day", 0.0)
    if bg_rate > 0:
        bg_rng = random.Random(scenario["seed"])
        engine.generate_background_episodes(
            start_ts, end_ts, bg_rng,
            episodes_per_signal_per_day=bg_rate,
            type_weights=scenario.get("type_weights"),
        )

    for ep in scenario.get("episodes", []):
        kwargs = {k: v for k, v in ep.items() if k not in ("station_id", "signal", "episode_type", "start_ts")}
        if kwargs.get("maintenance_ts"):
            kwargs["maintenance_ts"] = datetime.fromisoformat(kwargs["maintenance_ts"])
        engine.schedule_episode(
            ep["station_id"], ep["signal"], ep["episode_type"],
            datetime.fromisoformat(ep["start_ts"]), **kwargs,
        )

    injector = LatentDefectInjector(cfg.stations, engine, seed=scenario.get("defect_seed", 7))
    sim = LineSimulator(cfg, seed=scenario["seed"], signal_provider=engine, defect_injector=injector)
    sim.run(start_ts=start_ts, duration_s=duration_s)
    return sim, engine, start_ts, end_ts


def generate(scenario_name: str, out_dir: Path, line_config_path: str = "config/line.yaml",
             scenarios_config_path: str = "config/scenarios.yaml") -> dict:
    cfg = load_line_config(line_config_path)
    scenarios = load_scenarios(scenarios_config_path)
    if scenario_name not in scenarios:
        raise SystemExit(f"unknown scenario {scenario_name!r}; available: {sorted(scenarios)}")
    scenario = scenarios[scenario_name]

    sim, engine, start_ts, end_ts = build_scenario(cfg, scenario)

    events_df = sim.events_df()
    defects_df = sim.defects_df()
    surfaced_df = surface_defects(events_df, defects_df, cfg.stations,
                                   seed=scenario.get("defect_seed", 7) + 1)
    truth_df = pd.DataFrame(engine.truth_records())
    status_df = sim.status_df()

    out_dir.mkdir(parents=True, exist_ok=True)
    events_df.to_parquet(out_dir / "events.parquet", index=False)
    surfaced_df.to_parquet(out_dir / "ground_truth.parquet", index=False)
    truth_df.to_parquet(out_dir / "degradation_truth.parquet", index=False)
    status_df.to_parquet(out_dir / "status_intervals.parquet", index=False)

    return {
        "scenario": scenario_name, "out_dir": str(out_dir),
        "n_events": len(events_df), "n_defects": len(defects_df),
        "n_episodes": len(truth_df), "start_ts": start_ts, "end_ts": end_ts,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--out", default=None, help="output dir (default: data/<scenario>)")
    parser.add_argument("--line-config", default="config/line.yaml")
    parser.add_argument("--scenarios-config", default="config/scenarios.yaml")
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else Path("data") / args.scenario
    summary = generate(args.scenario, out_dir, args.line_config, args.scenarios_config)

    print(f"scenario={summary['scenario']} -> {summary['out_dir']}")
    print(f"window: {summary['start_ts']} .. {summary['end_ts']}")
    print(f"events: {summary['n_events']}  defects: {summary['n_defects']}  episodes: {summary['n_episodes']}")


if __name__ == "__main__":
    main()
