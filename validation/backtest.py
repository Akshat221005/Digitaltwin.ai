"""Validates models/forecast.py against simulator ground truth (BUILD_SPEC.md §6.4).

Replays every tier A/B signal reading chronologically through a fresh Forecaster,
then compares emitted forecasts against degradation_truth.parquet to produce:

  - forecast error vs. lead-time-remaining curve (median abs error + IQR, bucketed)
  - 95% CI coverage (fraction of true crossings inside the stated interval)
  - false forecast rate during noise_only episodes (target < 0.5/shift)
  - step-change refusal rate (target > 90%)

This module never looks at ground truth while forecasting — only afterward, to
grade it. That separation is load-bearing: it's what makes the numbers honest.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Optional

import pandas as pd

from simulator.line import LineConfig
from models.forecast import Forecaster, STATUS_FORECASTABLE

LEAD_TIME_BUCKET_EDGES_H = [1, 2, 4, 6, 8]  # ascending; a forecast lands in the
                                             # smallest edge >= its true remaining lead time


def replay_forecasts(events_df: pd.DataFrame, line_cfg: LineConfig) -> pd.DataFrame:
    """Runs a fresh Forecaster over every tier A/B signal reading in events_df,
    in chronological order per (station, signal). Returns one row per forecast
    emitted (i.e. per signal reading, once n_samples clears the minimum)."""
    forecaster = Forecaster(line_cfg.stations)
    records = []
    for station_id, signal in forecaster.instrumented_pairs():
        if signal not in events_df.columns:
            continue
        sub = events_df[(events_df.station_id == station_id) & events_df[signal].notna()]
        sub = sub.sort_values("entry_ts")
        for _, row in sub.iterrows():
            fc = forecaster.update(station_id, signal, row["entry_ts"], row[signal])
            ci = fc.hours_to_limit_ci or (None, None)
            records.append({
                "station_id": station_id, "signal": signal, "ts": row["entry_ts"],
                "status": fc.status, "hours_to_limit": fc.hours_to_limit,
                "hours_to_limit_ci_lo": ci[0], "hours_to_limit_ci_hi": ci[1],
                "reason": fc.reason, "confidence": fc.confidence, "n_samples": fc.n_samples,
            })
    return pd.DataFrame(records)


def _episode_window(truth_df: pd.DataFrame, ep: pd.Series, run_end_ts) -> tuple:
    """[start, end) during which this episode is the active one for its (station, signal)."""
    same_signal = truth_df[
        (truth_df.station_id == ep.station_id) & (truth_df.signal == ep.signal)
        & (truth_df.start_ts > ep.start_ts)
    ].sort_values("start_ts")
    end = same_signal.iloc[0]["start_ts"] if len(same_signal) else run_end_ts
    return ep.start_ts, end


def forecast_error_vs_lead_time(forecasts_df: pd.DataFrame, truth_df: pd.DataFrame) -> pd.DataFrame:
    """For every linear_wear/accelerating_wear episode with a real crossing, join
    every 'forecastable' forecast emitted during that episode's window against
    the true crossing time. One row per (episode, forecast-tick)."""
    episodes = truth_df[
        truth_df.episode_type.isin(["linear_wear", "accelerating_wear"])
        & truth_df.actual_spec_crossing_ts.notna()
    ]
    rows = []
    for _, ep in episodes.iterrows():
        actual = ep["actual_spec_crossing_ts"]
        window_start, window_end = ep["start_ts"], actual + timedelta(hours=1)
        sub = forecasts_df[
            (forecasts_df.station_id == ep.station_id) & (forecasts_df.signal == ep.signal)
            & (forecasts_df.ts >= window_start) & (forecasts_df.ts <= window_end)
            & (forecasts_df.status == STATUS_FORECASTABLE) & forecasts_df.hours_to_limit.notna()
        ]
        for _, f in sub.iterrows():
            predicted = f.ts + timedelta(hours=f.hours_to_limit)
            error_h = (predicted - actual).total_seconds() / 3600.0
            lead_h = (actual - f.ts).total_seconds() / 3600.0
            covered = None
            if pd.notna(f.hours_to_limit_ci_lo) and pd.notna(f.hours_to_limit_ci_hi):
                pred_lo = f.ts + timedelta(hours=f.hours_to_limit_ci_lo)
                pred_hi = f.ts + timedelta(hours=f.hours_to_limit_ci_hi)
                covered = bool(pred_lo <= actual <= pred_hi)
            rows.append({
                "episode_id": ep.episode_id, "episode_type": ep.episode_type,
                "station_id": ep.station_id, "signal": ep.signal, "ts": f.ts,
                "error_h": error_h, "lead_h": lead_h, "covered": covered,
            })
    return pd.DataFrame(rows)


def bucket_lead_time(lead_h: float) -> Optional[int]:
    for edge in LEAD_TIME_BUCKET_EDGES_H:
        if lead_h <= edge:
            return edge
    return None  # beyond the largest bucket — not reported, matches the 72h horizon gate


def lead_time_error_curve(joined_df: pd.DataFrame) -> pd.DataFrame:
    if joined_df.empty:
        return pd.DataFrame(columns=["bucket_h", "n", "median_abs_error_h", "iqr_lo_h", "iqr_hi_h"])
    df = joined_df.copy()
    df["bucket_h"] = df["lead_h"].apply(bucket_lead_time)
    df = df[df.bucket_h.notna()]
    df["abs_error_h"] = df["error_h"].abs()
    rows = []
    for bucket, g in df.groupby("bucket_h"):
        rows.append({
            "bucket_h": bucket, "n": len(g),
            "median_abs_error_h": float(g.abs_error_h.median()),
            "iqr_lo_h": float(g.abs_error_h.quantile(0.25)),
            "iqr_hi_h": float(g.abs_error_h.quantile(0.75)),
        })
    return pd.DataFrame(rows).sort_values("bucket_h")


def ci_coverage(joined_df: pd.DataFrame) -> float:
    covered = joined_df["covered"].dropna()
    if covered.empty:
        return float("nan")
    return float(covered.mean())


def false_forecast_rate(forecasts_df: pd.DataFrame, truth_df: pd.DataFrame, run_end_ts) -> dict:
    """Forecasts emitted (status == forecastable) during noise_only episode windows,
    normalized to an 8h shift."""
    noise_eps = truth_df[truth_df.episode_type == "noise_only"]
    total_false_episodes = 0
    total_hours = 0.0
    for _, ep in noise_eps.iterrows():
        start, end = _episode_window(truth_df, ep, run_end_ts)
        total_hours += (end - start).total_seconds() / 3600.0
        sub = forecasts_df[
            (forecasts_df.station_id == ep.station_id) & (forecasts_df.signal == ep.signal)
            & (forecasts_df.ts >= start) & (forecasts_df.ts < end)
            & (forecasts_df.status == STATUS_FORECASTABLE)
        ].sort_values("ts")
        if sub.empty:
            continue
        # count distinct episodes of forecastable status (transitions), not every tick
        prev_ts = None
        gap = timedelta(minutes=10)
        for ts in sub["ts"]:
            if prev_ts is None or (ts - prev_ts) > gap:
                total_false_episodes += 1
            prev_ts = ts
    shifts = total_hours / 8.0 if total_hours > 0 else float("nan")
    rate = total_false_episodes / shifts if shifts else float("nan")
    return {"false_episodes": total_false_episodes, "shifts_observed": shifts, "rate_per_shift": rate}


def step_change_refusal_rate(forecasts_df: pd.DataFrame, truth_df: pd.DataFrame, run_end_ts) -> dict:
    """Fraction of step_change episodes during which the forecaster never emits a
    confident (forecastable) crossing prediction over the episode's whole active
    window. Any abstention reason counts (explicit step-change detection,
    insufficient samples, residual spread) — the requirement is "never confidently
    predicts a wear trend through a step", not a specific gate reason or timing."""
    step_eps = truth_df[truth_df.episode_type == "step_change"]
    if step_eps.empty:
        return {"n_episodes": 0, "n_refused": 0, "rate": float("nan")}
    refused = 0
    for _, ep in step_eps.iterrows():
        start, end = _episode_window(truth_df, ep, run_end_ts)
        sub = forecasts_df[
            (forecasts_df.station_id == ep.station_id) & (forecasts_df.signal == ep.signal)
            & (forecasts_df.ts >= start) & (forecasts_df.ts < end)
            & (forecasts_df.status == STATUS_FORECASTABLE)
        ]
        if sub.empty:
            refused += 1
    return {"n_episodes": len(step_eps), "n_refused": refused, "rate": refused / len(step_eps)}


def run_backtest(events_df: pd.DataFrame, truth_df: pd.DataFrame, line_cfg: LineConfig, run_end_ts) -> dict:
    forecasts_df = replay_forecasts(events_df, line_cfg)
    joined = forecast_error_vs_lead_time(forecasts_df, truth_df)
    return {
        "forecasts_df": forecasts_df,
        "joined_df": joined,
        "lead_time_curve": lead_time_error_curve(joined),
        "ci_coverage": ci_coverage(joined),
        "false_forecast": false_forecast_rate(forecasts_df, truth_df, run_end_ts),
        "step_change_refusal": step_change_refusal_rate(forecasts_df, truth_df, run_end_ts),
    }


def print_report(results: dict) -> None:
    print("=== Forecast error vs. lead time ===")
    print(results["lead_time_curve"].to_string(index=False))
    print(f"\n=== CI coverage (target ~95%) ===\n{results['ci_coverage']:.1%}")
    ff = results["false_forecast"]
    print(f"\n=== False forecast rate (target < 0.5/shift) ===\n"
          f"{ff['false_episodes']} episodes over {ff['shifts_observed']:.1f} shifts "
          f"= {ff['rate_per_shift']:.2f}/shift")
    sc = results["step_change_refusal"]
    print(f"\n=== Step-change refusal rate (target > 90%) ===\n"
          f"{sc['n_refused']}/{sc['n_episodes']} = {sc['rate']:.1%}")
