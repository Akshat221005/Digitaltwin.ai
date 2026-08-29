"""Supervisor tab: real-time line map, detections, degradation forecasts,
at-risk vehicles, and recommendation cards (BUILD_SPEC.md §10)."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from datetime import timedelta

from app.recommendations import (
    STATUS_ACKNOWLEDGED, STATUS_EXECUTED, STATUS_PROPOSED,
    detection_recommendation, forecast_recommendation,
)
from models.cascade import run_single_cascade
from models.forecast import STATUS_CROSSED, STATUS_DRIFTING, STATUS_FORECASTABLE
from twin.state import current_station_status

DETECTION_RECENCY_MINUTES = 15  # how long a detection counts as "still active" with no newer reading

# Cascade projection assumptions for "if unaddressed" — see ASSUMPTIONS.md.
UNPLANNED_DOWN_DURATION_S = 45 * 60   # unplanned failure repair time once truly crossed
CASCADE_HORIZON_S = 4 * 3600          # how far forward to project consequences

STATUS_COLORS = {
    "working": "#2ecc71", "blocked": "#f39c12", "starved": "#95a5a6", "down": "#e74c3c",
}
TIER_MARKERS = {"A": "diamond", "B": "square", "C": "circle"}
STATUS_BADGE = {
    STATUS_PROPOSED: "🟡 Proposed", STATUS_ACKNOWLEDGED: "🔵 Acknowledged",
    STATUS_EXECUTED: "✅ Executed by human",
}


def _line_map(cfg, status_now: dict, projected_affected: set = None) -> go.Figure:
    projected_affected = projected_affected or set()
    fig = go.Figure()
    x = 0
    zone_bounds = {}
    for zone in cfg.zones:
        start_x = x
        for s in [s for s in cfg.stations if s.zone == zone]:
            color = STATUS_COLORS.get(status_now.get(s.id, "starved"), "#bdc3c7")
            projected = s.id in projected_affected
            fig.add_trace(go.Scatter(
                x=[x], y=[0], mode="markers+text",
                text=[f"⚠ {s.id}" if projected else s.id], textposition="top center",
                marker=dict(
                    size=24 if projected else 20, color=color, symbol=TIER_MARKERS[s.sensor_tier],
                    line=dict(width=4 if projected else 1, color="#c0392b" if projected else "#333"),
                ),
                hovertext=(
                    f"{s.id} ({s.name}) — tier {s.sensor_tier} — {status_now.get(s.id, '?')}"
                    + (" — PROJECTED to block/starve if unaddressed" if projected else "")
                ),
                hoverinfo="text", showlegend=False,
            ))
            x += 1
        zone_bounds[zone] = (start_x, x - 1)
    for zone, (lo, hi) in zone_bounds.items():
        fig.add_annotation(x=(lo + hi) / 2, y=-0.4, text=zone, showarrow=False, font=dict(size=11, color="#888"))
    fig.update_layout(height=170, xaxis=dict(visible=False), yaxis=dict(visible=False, range=[-1, 1]),
                       margin=dict(l=10, r=10, t=10, b=10))
    return fig


AFFECTED_EXTRA_DISRUPTION_S = 180  # threshold above baseline to count as "affected"


def _disrupted_seconds_by_station(status_df, window_start):
    df = status_df[status_df.end_ts > window_start]
    df = df[df.status.isin(["blocked", "starved", "down"])]
    if df.empty:
        return df.assign(dur=[])["dur"] if "dur" in df.columns else df.assign(dur=0).dur
    dur = (df.end_ts - df.start_ts).dt.total_seconds()
    return df.assign(dur=dur).groupby("station_id").dur.sum()


def _projected_affected_stations(cascade_result, baseline_result, station_id: str) -> set:
    """Stations meaningfully MORE blocked/starved/down in the failure fork than
    they would have been anyway (compared against a same-horizon fork with no
    failure) — not just any station that ever blips blocked/starved, which
    happens routinely from ordinary buffer dynamics even with no failure at
    all. Ordinary small-buffer JIT lines really do cascade a stoppage across
    most of the downstream chain within a few hours — a wide affected set here
    can be a genuine finding, not a bug; the baseline comparison is what makes
    that trustworthy instead of noise."""
    dn = _disrupted_seconds_by_station(cascade_result.status_df, cascade_result.failure_ts)
    bl = _disrupted_seconds_by_station(baseline_result.status_df, baseline_result.failure_ts)
    all_ids = set(dn.index) | set(bl.index)
    affected = {
        sid for sid in all_ids
        if dn.get(sid, 0.0) - bl.get(sid, 0.0) > AFFECTED_EXTRA_DISRUPTION_S
    }
    affected.add(station_id)
    return affected


def render(state, cfg, get_live_sim=None):
    st.subheader("Line status")

    # A crossed forecast is a confirmed physical fact — the station really is
    # out of spec — but the historical replay data was generated assuming
    # nobody ever intervenes AND that drift alone never stops the line. If
    # that's still true "as of now", simulate what actually happens next
    # (models/cascade.py) rather than just showing a static forecast number
    # with no visible consequence on the line.
    crossed = [(k, fc) for k, fc in state.forecasts.items() if fc.status == STATUS_CROSSED]
    projected_affected = set()
    cascade_result = None
    if crossed and get_live_sim is not None:
        (station_id, signal), fc = crossed[0]
        try:
            sim = get_live_sim()
            cascade_result = run_single_cascade(
                sim, station_id, failure_ts=state.as_of,
                down_duration_s=UNPLANNED_DOWN_DURATION_S, horizon_s=CASCADE_HORIZON_S,
                label="do_nothing",
            )
            # Same-horizon fork with (effectively) no failure, so "affected" means
            # meaningfully worse than what would have happened anyway — not just
            # any station that blips blocked/starved from ordinary buffer dynamics.
            baseline_result = run_single_cascade(
                sim, station_id, failure_ts=state.as_of,
                down_duration_s=0, horizon_s=CASCADE_HORIZON_S, label="baseline",
            )
            projected_affected = _projected_affected_stations(cascade_result, baseline_result, station_id)
        except Exception as e:
            st.warning(f"Cascade projection unavailable: {e}")

    status_now = current_station_status(state.status, state.as_of)
    st.plotly_chart(_line_map(cfg, status_now, projected_affected), use_container_width=True)
    st.caption(
        "🟢 working · 🟠 blocked · ⬜ starved · 🔴 down &nbsp;|&nbsp; "
        "marker shape: ◆ tier A (rich) ■ tier B (partial) ● tier C (dark, barcode only) &nbsp;|&nbsp; "
        "⚠ red outline = projected to block/starve if unaddressed (simulated, not yet observed)",
        unsafe_allow_html=True,
    )

    if cascade_result is not None:
        starv_txt = (
            cascade_result.time_to_first_starvation.strftime("%H:%M")
            if cascade_result.time_to_first_starvation is not None else "no starvation projected in the window"
        )
        st.error(
            f"**Simulation — if unaddressed:** {station_id} {signal} is confirmed at/beyond spec. "
            f"Projecting an unplanned {UNPLANNED_DOWN_DURATION_S // 60}-min failure from now: "
            f"first downstream starvation **{starv_txt}**, "
            f"stoppage **{cascade_result.stoppage_duration_s / 60:.0f} min**, "
            f"**{len(projected_affected) - 1}** other station(s) affected, "
            f"**${cascade_result.rework_cost:,.0f}** rework cost from defects surfacing in the window. "
            f"This is a simulated projection, not an observed fact — nothing has actually happened yet."
        )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Active detections")
        st.caption("An anomaly happening *now* — never confused with a forecast.")
        if not state.detections:
            st.info("No active detections.")
        else:
            for d in sorted(state.detections, key=lambda d: d.ts, reverse=True)[:10]:
                if not d.confirmed:
                    icon = "⚪"  # single-tick, not yet persisted — visible, but not actionable
                else:
                    icon = "🔴" if d.severity == "critical" else "🟠"
                st.markdown(f"{icon} **{d.station_id} {d.signal}** — {d.reason}  \n"
                            f"value={d.value:.2f} · {d.ts.strftime('%H:%M:%S')}")

    with col2:
        st.subheader("Degradation forecasts")
        st.caption("The only true *prediction* in this system.")
        active = {k: v for k, v in state.forecasts.items()
                  if v.status in (STATUS_FORECASTABLE, STATUS_DRIFTING, STATUS_CROSSED)}
        if not active:
            st.success("All monitored signals stable — no forecasts in progress.")
        for (station_id, signal), fc in active.items():
            conf_level = "High" if fc.confidence >= 0.75 else ("Medium" if fc.confidence >= 0.45 else "Low")
            if fc.status == STATUS_FORECASTABLE:
                st.markdown(
                    f"**{station_id} {signal} — {fc.hours_to_limit:.1f}h to tolerance limit** "
                    f"(range {fc.hours_to_limit_ci[0]:.1f}–{fc.hours_to_limit_ci[1]:.1f}h)"
                )
                st.caption(f"{fc.reason}. Confidence: **{conf_level}** "
                           f"({fc.n_samples} samples, tier {state.tier_of.get(station_id)}).")
            elif fc.status == STATUS_CROSSED:
                st.markdown(f"🔴 **{station_id} {signal} — already at/beyond spec limit**")
            else:
                st.markdown(f"**{station_id} {signal} — {fc.status}**")
                st.caption(fc.reason)

    st.subheader("At-risk vehicles")
    st.caption("A latent condition that already exists but hasn't surfaced yet.")
    if state.at_risk.empty:
        st.info("No vehicles currently flagged at risk.")
    else:
        show = state.at_risk[[
            "vin", "variant", "origin_station", "last_known_station",
            "expected_surfacing_station", "eta", "risk_score", "confidence",
        ]]
        st.dataframe(show, use_container_width=True, hide_index=True)
        st.caption(f"{len(show)} vehicles carrying elevated risk downstream.")

    st.subheader("Recommendations")
    st.caption("Proposed → Acknowledged → Executed by human. This twin never writes to a PLC.")
    if "recommendations" not in st.session_state:
        st.session_state.recommendations = {}
    store = st.session_state.recommendations

    # Most recent CONFIRMED detection per (station, signal) — an unconfirmed
    # single-tick spike is shown above but must never, on its own, drive a
    # recommendation card. A detection that hasn't had a newer confirmed
    # reading in a while doesn't count as "still active" either.
    latest_detection = {}
    for d in state.detections:
        if not d.confirmed:
            continue
        key = (d.station_id, d.signal)
        if key not in latest_detection or d.ts > latest_detection[key].ts:
            latest_detection[key] = d
    recency_cutoff = state.as_of - timedelta(minutes=DETECTION_RECENCY_MINUTES)
    active_detections = {
        key: d for key, d in latest_detection.items() if d.ts >= recency_cutoff
    }

    active_keys = set()
    all_pairs = set(state.forecasts.keys()) | set(active_detections.keys())
    for station_id, signal in all_pairs:
        fc = state.forecasts.get((station_id, signal))
        rec_key = f"{station_id}:{signal}"

        if fc is not None and fc.status in (STATUS_FORECASTABLE, STATUS_DRIFTING, STATUS_CROSSED):
            # Forecast is confident enough to commit to a countdown — this
            # supersedes the plain "investigate now" detection card.
            fresh = forecast_recommendation(station_id, signal, fc, state.as_of)
        elif (station_id, signal) in active_detections:
            # Something is happening now, but the forecaster hasn't accumulated
            # enough consecutive significant readings yet to predict a crossing
            # time — don't make the supervisor wait for that to be told to look.
            fresh = detection_recommendation(
                station_id, signal, active_detections[(station_id, signal)], state.as_of, fc,
            )
        else:
            continue

        active_keys.add(rec_key)
        fresh.id = rec_key
        if rec_key in store:
            store[rec_key].title, store[rec_key].detail = fresh.title, fresh.detail
        else:
            store[rec_key] = fresh

    # A card only exists while its underlying detection/forecast is still
    # active as of `state.as_of`. Without this, dragging the time slider
    # backward (or past a drift and back) left stale cards on screen —
    # recommendations from a state the twin, at this point in time, hasn't
    # actually reached.
    for stale_key in [k for k in store if k not in active_keys]:
        del store[stale_key]

    if not store:
        st.caption("No recommendations pending.")
    for key, rec in list(store.items()):
        with st.container(border=True):
            st.markdown(f"**{rec.title}**")
            st.caption(rec.detail)
            st.markdown(STATUS_BADGE[rec.status])
            c1, _ = st.columns(2)
            if rec.status == STATUS_PROPOSED:
                if c1.button("Acknowledge", key=f"ack_{key}"):
                    rec.advance()
                    st.rerun()
            elif rec.status == STATUS_ACKNOWLEDGED:
                if c1.button("Mark executed by human", key=f"exec_{key}"):
                    rec.advance()
                    st.rerun()
