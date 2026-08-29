"""Writes validation metrics + plots to /docs (BUILD_SPEC.md §13's required
validation results table comes straight from this module's output).

Run: python -m validation.report
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px

from models.infer import infer_at_risk_vehicles
from simulator.line import load_line_config
from twin.confidence import TIER_CONFIDENCE
from twin.genealogy import Genealogy
from validation.backtest import run_backtest

DOCS_DIR = Path("docs")


def _load(scenario: str):
    base = Path("data") / scenario
    events = pd.read_parquet(base / "events.parquet")
    truth = pd.read_parquet(base / "degradation_truth.parquet")
    ground_truth = pd.read_parquet(base / "ground_truth.parquet")
    return events, truth, ground_truth


def forecaster_validation(scenario: str = "noisy_line") -> dict:
    """The headline forecaster metrics (§6.4), measured against the scenario
    with the richest mix of episode types."""
    events, truth, _ = _load(scenario)
    cfg = load_line_config("config/line.yaml")
    run_end_ts = events.exit_ts.max() + timedelta(minutes=1)
    results = run_backtest(events, truth, cfg, run_end_ts)

    curve = results["lead_time_curve"]
    fig = px.bar(
        curve, x="bucket_h", y="median_abs_error_h", error_y=curve.iqr_hi_h - curve.median_abs_error_h,
        error_y_minus=curve.median_abs_error_h - curve.iqr_lo_h,
        labels={"bucket_h": "Lead time remaining (h, bucketed)", "median_abs_error_h": "Median abs. forecast error (h)"},
        title=f"Forecast error vs. lead time remaining — {scenario} scenario (error shrinks as crossing nears)",
    )
    DOCS_DIR.mkdir(exist_ok=True)
    fig.write_html(DOCS_DIR / "forecast_error_vs_leadtime.html", include_plotlyjs="cdn")

    return {
        "scenario": scenario,
        "lead_time_curve": curve.to_dict(orient="records"),
        "ci_coverage": results["ci_coverage"],
        "false_forecast_rate_per_shift": results["false_forecast"]["rate_per_shift"],
        "step_change_refusal_rate": results["step_change_refusal"]["rate"],
        "n_step_change_episodes": results["step_change_refusal"]["n_episodes"],
    }


def defect_inference_precision_recall(scenario: str = "weld_drift_demo",
                                       snapshot_ts: datetime = None) -> dict:
    """Precision/recall of the at-risk VIN inference (§7.3), measured against
    ground truth: of the VINs flagged at-risk from the S12 drift window, what
    fraction actually carried a real (ground-truth) defect, and what fraction
    of real defects did the inference catch?"""
    events, _, ground_truth = _load(scenario)
    cfg = load_line_config("config/line.yaml")

    drift_ep = pd.read_parquet(Path("data") / scenario / "degradation_truth.parquet").iloc[0]
    window_start = drift_ep["start_ts"]
    # Snapshot at the moment the signal actually crosses spec — a realistic "flag
    # them now" moment (matching the demo narrative), not the end of the run.
    # Querying at end-of-run instead measures only the handful of vehicles still
    # in transit when the simulation happens to stop, which is a near-arbitrary
    # trailing cohort and not representative of how a supervisor would use this.
    snapshot_ts = snapshot_ts or drift_ep["actual_spec_crossing_ts"]

    live_events = events[events.exit_ts <= snapshot_ts]
    genealogy = Genealogy(live_events, cfg)
    zone_of = {s.id: s.zone for s in cfg.stations}

    at_risk = infer_at_risk_vehicles(
        genealogy, drift_ep["station_id"], drift_ep["signal"], window_start, snapshot_ts,
        deviation_pct=5.0, zone_of=zone_of, sensor_tier="A", tier_confidence=TIER_CONFIDENCE,
    )
    flagged_vins = {v.vin for v in at_risk}

    # Ground truth over the SAME window: real defects among vehicles that passed
    # the drifting station before the snapshot (matching what the flagging could
    # possibly have caught), not the whole scenario's defects.
    window_events = events[(events.station_id == drift_ep["station_id"])
                            & (events.entry_ts >= window_start) & (events.entry_ts <= snapshot_ts)]
    eligible_vins = set(window_events.vin)
    real_defect_vins = set(
        ground_truth[(ground_truth.injected_at_station == drift_ep["station_id"])
                      & ground_truth.vin.isin(eligible_vins)].vin
    )
    tp = len(flagged_vins & real_defect_vins)
    precision = tp / len(flagged_vins) if flagged_vins else float("nan")
    recall = tp / len(real_defect_vins) if real_defect_vins else float("nan")

    return {
        "scenario": scenario, "snapshot_ts": str(snapshot_ts), "n_flagged": len(flagged_vins),
        "n_real_defects": len(real_defect_vins), "true_positives": tp,
        "precision": precision, "recall": recall,
    }


def detection_lead_time(scenario: str = "weld_drift_demo") -> dict:
    """Mean lead time: detection timestamp vs. the moment a defect first
    surfaces at end-of-line — the '78 minutes earlier than EOL testing' claim,
    measured rather than narrated."""
    from models.detect import Detector

    events, truth, ground_truth = _load(scenario)
    ep = truth.iloc[0]
    cfg = load_line_config("config/line.yaml")
    detector = Detector(cfg.stations)

    sub = events[(events.station_id == ep.station_id) & events[ep.signal].notna()].sort_values("entry_ts")
    first_detection_ts = None
    for row in sub.itertuples():
        hits = detector.update(ep.station_id, ep.signal, getattr(row, ep.signal), row.entry_ts)
        if hits:
            first_detection_ts = row.entry_ts
            break

    surfaced = ground_truth[ground_truth.injected_at_station == ep.station_id]
    surfaced = surfaced[surfaced.surfaced_ts.notna()].sort_values("surfaced_ts")
    first_surfacing_ts = surfaced.iloc[0]["surfaced_ts"] if len(surfaced) else None

    lead_minutes = None
    if first_detection_ts is not None and first_surfacing_ts is not None:
        lead_minutes = (first_surfacing_ts - first_detection_ts).total_seconds() / 60.0

    return {
        "scenario": scenario, "first_detection_ts": str(first_detection_ts),
        "first_surfacing_ts": str(first_surfacing_ts), "lead_time_minutes": lead_minutes,
    }


def write_report():
    DOCS_DIR.mkdir(exist_ok=True)
    forecast_metrics = forecaster_validation("noisy_line")
    inference_metrics = defect_inference_precision_recall("weld_drift_demo")
    lead_time = detection_lead_time("weld_drift_demo")

    all_metrics = {
        "forecaster": forecast_metrics,
        "defect_inference": inference_metrics,
        "detection_lead_time": lead_time,
    }
    with open(DOCS_DIR / "validation_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    lines = ["# Validation results\n"]
    lines.append("## Degradation forecaster (measured on the `noisy_line` scenario)\n")
    lines.append("| Lead time bucket (h) | n | Median abs. error (h) | IQR |")
    lines.append("|---|---|---|---|")
    for row in forecast_metrics["lead_time_curve"]:
        lines.append(f"| {row['bucket_h']:g} | {row['n']} | {row['median_abs_error_h']:.2f} | "
                     f"{row['iqr_lo_h']:.2f}-{row['iqr_hi_h']:.2f} |")
    lines.append(f"\n- **95% CI coverage:** {forecast_metrics['ci_coverage']:.1%} (target ~95%)")
    lines.append(f"- **False forecast rate:** {forecast_metrics['false_forecast_rate_per_shift']:.2f}/shift (target < 0.5/shift)")
    lines.append(f"- **Step-change refusal rate:** {forecast_metrics['step_change_refusal_rate']:.1%} "
                 f"over {forecast_metrics['n_step_change_episodes']} episodes (target > 90%)")

    lines.append("\n## Defect inference (measured on the `weld_drift_demo` scenario)\n")
    lines.append(f"- Precision: {inference_metrics['precision']:.1%} "
                 f"({inference_metrics['true_positives']}/{inference_metrics['n_flagged']} flagged VINs had a real defect)")
    lines.append(f"- Recall: {inference_metrics['recall']:.1%} "
                 f"({inference_metrics['true_positives']}/{inference_metrics['n_real_defects']} real defects flagged)")

    lines.append("\n## Detection lead time (measured on the `weld_drift_demo` scenario)\n")
    lines.append(f"- First detection: {lead_time['first_detection_ts']}")
    lines.append(f"- First real-world surfacing (EOL inspection): {lead_time['first_surfacing_ts']}")
    if lead_time["lead_time_minutes"] is not None:
        lines.append(f"- **Lead time gained: {lead_time['lead_time_minutes']:.0f} minutes**")

    lines.append("\nInteractive plot: [forecast_error_vs_leadtime.html](forecast_error_vs_leadtime.html)")

    with open(DOCS_DIR / "validation_metrics.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    return all_metrics


if __name__ == "__main__":
    write_report()
