"""Plant manager tab: simplified, human-friendly overview of line health,
bottleneck analysis, rework trends, maintenance planning, and sensor ROI."""
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
        labels=dict(x="Hour of Day", y="Station", color="Time as Bottleneck (h)"),
        title="Where is the line being slowed down, and when?",
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),
        coloraxis_colorbar=dict(title="Hours")
    )
    return fig


def _rework_trend(ground_truth_df: pd.DataFrame, rework_cost: float):
    surfaced = ground_truth_df[ground_truth_df.surfaced_ts.notna()].copy()
    if surfaced.empty:
        return None
    surfaced["day"] = pd.to_datetime(surfaced.surfaced_ts).dt.floor("D")
    trend = surfaced.groupby("day").size().rename("n_defects").reset_index()
    trend["rework_cost"] = trend.n_defects * rework_cost
    fig = px.bar(
        trend, x="day", y="rework_cost",
        title="Daily cost of fixing defects found at end-of-line inspection",
        labels={"day": "Date", "rework_cost": "Rework Cost ($)"},
        color_discrete_sequence=["#ef4444"],
    )
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    return fig


def render(state, cfg, full_status_df, ground_truth_df):

    # OEE
    st.subheader("Line Availability (OEE)")
    st.markdown(
        "<div class='caption-text'>"
        "<b>What is this?</b> OEE (Overall Equipment Effectiveness) tells you what percentage of "
        "scheduled production time your factory line was actually <b>running and producing</b>. "
        "100% means the line was running perfectly. A lower number means time is being lost to "
        "breakdowns, starvation, or blocking."
        "</div>",
        unsafe_allow_html=True
    )
    oee = _oee(state.status, len(cfg.stations))
    val = oee["availability"]
    if val == val:
        color = "#22c55e" if val > 0.90 else ("#f59e0b" if val > 0.75 else "#ef4444")
        health = "Healthy" if val > 0.90 else ("Moderate — check bottlenecks" if val > 0.75 else "Low — immediate attention needed")
        st.markdown(
            f"<div style='font-size: 2.4rem; font-weight: 800; color: {color};'>{val:.1%}</div>"
            f"<div style='color:#64748b; font-size: 0.9em; margin-top:-6px;'>Line health: <b>{health}</b></div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown("<div style='font-size: 2.4rem; font-weight: 800; color: #94a3b8;'>N/A</div>", unsafe_allow_html=True)

    st.divider()

    # Bottleneck Analysis
    st.subheader("Bottleneck Analysis")
    st.markdown(
        "<div class='caption-text'>"
        "<b>What is this?</b> A bottleneck is the slowest station on the line — it dictates the pace of "
        "the entire factory. The heatmap below shows <b>which station was the bottleneck at each hour</b>. "
        "Darker orange = more time spent as the limiting factor."
        "</div>",
        unsafe_allow_html=True
    )
    migration = bottleneck_migration(state.status)
    fig = _bottleneck_heatmap(migration)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
        hours = bottleneck_hours_by_station(migration)
        if len(hours):
            top_stn = hours.index[0]
            top_h = hours.iloc[0]
            st.markdown(
                f"<div class='insight-box'><b>Key Insight:</b> Station <b>{top_stn}</b> was the primary bottleneck "
                f"for <b>{top_h:.1f} hours</b>. Fixing this station has the biggest impact on total throughput.</div>",
                unsafe_allow_html=True
            )
    else:
        st.info("Not enough data yet to compute a bottleneck heatmap.")

    st.divider()

    # Rework Cost Trend
    st.subheader("Defect Rework Cost Trend")
    st.markdown(
        "<div class='caption-text'>"
        "<b>What is this?</b> When a defective vehicle reaches the end of the line and is caught at inspection, "
        "it has to be sent back for rework. Each bar shows <b>how much that cost per day</b>. "
        "The digital twin aims to catch defects at the source — before they become expensive rework."
        "</div>",
        unsafe_allow_html=True
    )
    live_ground_truth = ground_truth_df[ground_truth_df.injected_ts <= state.as_of]
    rework_fig = _rework_trend(live_ground_truth, DEFAULT_REWORK_COST_PER_UNIT)
    if rework_fig is not None:
        st.plotly_chart(rework_fig, use_container_width=True)
    else:
        st.info("No defects have surfaced yet in this time window.")

    st.divider()

    # Maintenance Window Planner
    st.subheader("Maintenance Window Planner")
    st.markdown(
        "<div class='caption-text'>"
        "<b>What is this?</b> This table is built from the Degradation Forecasts. "
        "It calculates exactly when maintenance needs to happen and snaps it to the "
        "<b>next available shift change</b> to avoid unplanned mid-shift breakdowns."
        "</div>",
        unsafe_allow_html=True
    )
    active = {k: v for k, v in state.forecasts.items()
              if v.status in (STATUS_FORECASTABLE, STATUS_DRIFTING, STATUS_CROSSED)}
    if not active:
        st.success("No maintenance urgently required right now — all signals stable.")
    else:
        rows = []
        for (station_id, signal), fc in active.items():
            urgency = "Act Now" if fc.status == STATUS_CROSSED else (
                "This Shift" if (fc.hours_to_limit or 99) < 4 else "Plan Ahead"
            )
            rows.append({
                "Station": station_id,
                "Signal": signal,
                "Urgency": urgency,
                "Hours Remaining": f"{fc.hours_to_limit:.1f}h" if fc.hours_to_limit else "Crossed",
                "Service Before": next_shift_change(state.as_of).strftime("%H:%M"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.markdown(
            "<div class='insight-box'>Schedule maintenance at shift changes (shown above) "
            "to avoid disrupting active production runs.</div>",
            unsafe_allow_html=True
        )

    st.divider()

    # Sensor Retrofit Ranking
    st.subheader("Where Should We Install Sensors Next?")
    st.markdown(
        "<div class='caption-text'>"
        "<b>What is this?</b> Not every station has sensors. 'Dark' stations (Tier C) only scan barcodes "
        "— they have no process data like temperature or current. This ranking uses vehicle genealogy to "
        "figure out which dark station is most likely causing downstream defects, ranked by "
        "<b>bang-for-buck</b> (defects prevented per dollar of sensor installation)."
        "</div>",
        unsafe_allow_html=True
    )
    retrofit_cost = default_retrofit_cost_by_station(cfg.stations)
    ranking = retrofit_ranking(live_ground_truth, cfg.stations, DEFAULT_REWORK_COST_PER_UNIT, retrofit_cost)
    if ranking.empty:
        st.info("No dark-station defects attributed yet in this time window.")
    else:
        st.dataframe(ranking.head(10), use_container_width=True, hide_index=True)
        top = ranking.iloc[0]
        st.markdown(
            f"<div class='insight-box'><b>Top Recommendation:</b> Install sensors at station "
            f"<b>{top.station_id}</b> next — linked to <b>{int(top.defect_count)} defects</b> "
            f"via vehicle genealogy. Estimated retrofit cost: <b>${top.retrofit_cost:,.0f}</b>.</div>",
            unsafe_allow_html=True
        )
