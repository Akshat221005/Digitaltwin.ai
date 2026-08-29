"""Plant manager tab: OEE, bottleneck migration heatmap, rework cost trend,
maintenance window planner, sensor retrofit ranking (BUILD_SPEC.md §10)."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app.recommendations import next_shift_change
from models.bottleneck import bottleneck_hours_by_station, bottleneck_migration
from models.forecast import STATUS_CROSSED, STATUS_DRIFTING, STATUS_FORECASTABLE
from twin.confidence import (
    DEFAULT_REWORK_COST_PER_UNIT, default_retrofit_cost_by_station, retrofit_ranking,
)


def _oee(status_df: pd.DataFrame, n_stations: int) -> dict:
    """Simplified OEE: availability = working time / total logged time (across all
    stations); performance and quality are not separately modelled here (would
    need a takt-time-vs-actual comparison and a full quality-pass definition
    beyond what this prototype tracks) — flagged as a limitation in the README."""
    if status_df.empty:
        return {"availability": float("nan")}
    dur_s = (status_df.end_ts - status_df.start_ts).dt.total_seconds()
    total_s = dur_s.sum()
    working_s = dur_s[status_df.status == "working"].sum()
    availability = working_s / total_s if total_s else float("nan")
    return {"availability": availability}


def _bottleneck_heatmap(migration_df: pd.DataFrame):
    if migration_df.empty:
        return None
    df = migration_df.copy()
    df["dur_h"] = (df.end_ts - df.start_ts).dt.total_seconds() / 3600.0
    df["hour"] = df.start_ts.dt.floor("h")
    pivot = df.groupby(["station_id", "hour"]).dur_h.sum().unstack(fill_value=0)
    fig = px.imshow(
        pivot, aspect="auto", color_continuous_scale="Oranges",
        labels=dict(x="Hour", y="Station", color="Bottleneck hours"),
        title="Bottleneck migration (station × hour) — proves the bottleneck moves",
    )
    return fig


def _rework_trend(ground_truth_df: pd.DataFrame, rework_cost: float):
    surfaced = ground_truth_df[ground_truth_df.surfaced_ts.notna()].copy()
    if surfaced.empty:
        return None
    surfaced["day"] = pd.to_datetime(surfaced.surfaced_ts).dt.floor("D")
    trend = surfaced.groupby("day").size().rename("n_defects").reset_index()
    trend["rework_cost"] = trend.n_defects * rework_cost
    fig = px.bar(trend, x="day", y="rework_cost", title="Rework cost trend (surfaced defects)")
    return fig


def render(state, cfg, full_status_df, ground_truth_df):
    st.subheader("Weekly OEE (simplified)")
    oee = _oee(state.status, len(cfg.stations))
    st.metric("Availability (working time / total time)", f"{oee['availability']:.1%}" if oee["availability"] == oee["availability"] else "n/a")
    st.caption("Performance and quality components are not modelled in this prototype — see README limitations.")

    st.subheader("Bottleneck migration heatmap")
    migration = bottleneck_migration(state.status)
    fig = _bottleneck_heatmap(migration)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data yet to compute a bottleneck heatmap.")
    hours = bottleneck_hours_by_station(migration)
    if len(hours):
        st.caption(f"Momentary bottleneck so far: **{hours.index[0]}** ({hours.iloc[0]:.1f}h).")

    st.subheader("Rework cost trend")
    live_ground_truth = ground_truth_df[ground_truth_df.injected_ts <= state.as_of]
    rework_fig = _rework_trend(live_ground_truth, DEFAULT_REWORK_COST_PER_UNIT)
    if rework_fig is not None:
        st.plotly_chart(rework_fig, use_container_width=True)
    else:
        st.info("No surfaced defects yet.")

    st.subheader("Maintenance window planner")
    st.caption("Forecast horizons snapped to the next scheduled shift change.")
    active = {k: v for k, v in state.forecasts.items()
              if v.status in (STATUS_FORECASTABLE, STATUS_DRIFTING, STATUS_CROSSED)}
    if not active:
        st.info("No maintenance windows currently recommended.")
    else:
        rows = []
        for (station_id, signal), fc in active.items():
            rows.append({
                "station": station_id, "signal": signal, "status": fc.status,
                "hours_to_limit": fc.hours_to_limit,
                "recommended_service_by": next_shift_change(state.as_of),
            })
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    st.subheader("Sensor retrofit ranking")
    st.caption(
        "Which tier-C (dark) stations to instrument next — ranked by defects "
        "back-attributed via genealogy, weighted by rework cost, per retrofit dollar."
    )
    retrofit_cost = default_retrofit_cost_by_station(cfg.stations)
    ranking = retrofit_ranking(live_ground_truth, cfg.stations, DEFAULT_REWORK_COST_PER_UNIT, retrofit_cost)
    if ranking.empty:
        st.info("No dark-station defects attributed yet.")
    else:
        st.dataframe(ranking.head(10), width='stretch', hide_index=True)
        top = ranking.iloc[0]
        st.caption(
            f"Suggested: instrument **{top.station_id}** next — "
            f"{int(top.defect_count)} defects attributed, ${top.retrofit_cost:,.0f} retrofit cost."
        )
