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
    STATUS_PROPOSED: "<span class='badge-proposed'>PROPOSED</span>",
    STATUS_ACKNOWLEDGED: "<span class='badge-acknowledged'>ACKNOWLEDGED</span>",
    STATUS_EXECUTED: "<span class='badge-executed'>EXECUTED</span>",
}


def _line_map(cfg, status_now: dict, projected_affected: set = None) -> go.Figure:
    projected_affected = projected_affected or set()
    fig = go.Figure()
    
    # We will lay out the zones in rows, flowing left to right, then moving down.
    num_zones = len(cfg.zones)
    y_coords = {zone: num_zones - i - 1 for i, zone in enumerate(cfg.zones)}
    
    xs, ys = [], []
    for zone in cfg.zones:
        y = y_coords[zone]
        x = 0
        zone_stations = [s for s in cfg.stations if s.zone == zone]
        
        # Draw background connecting line for the zone
        fig.add_trace(go.Scatter(
            x=[0, len(zone_stations)-1], y=[y, y], mode="lines",
            line=dict(color="#cbd5e1", width=3), hoverinfo="skip", showlegend=False
        ))
        
        for s in zone_stations:
            color = STATUS_COLORS.get(status_now.get(s.id, "starved"), "#bdc3c7")
            projected = s.id in projected_affected
            
            # Neater symbols: rounded squares, larger sizes, like the reference
            symbol = "square" if s.sensor_tier == "B" else ("diamond" if s.sensor_tier == "A" else "circle")
            
            fig.add_trace(go.Scatter(
                x=[x], y=[y], mode="markers+text",
                text=[f"⚠ {s.id}" if projected else s.id], textposition="top center",
                marker=dict(
                    size=28 if projected else 24, color=color, symbol=symbol,
                    line=dict(width=4 if projected else 2, color="#ef4444" if projected else "#475569"),
                ),
                hovertext=(
                    f"<b>{s.id} ({s.name})</b><br>"
                    f"Tier: {s.sensor_tier} <br>"
                    f"Status: {status_now.get(s.id, '?').upper()}"
                    + ("<br><span style='color:red;'>⚠ PROJECTED IMPACT</span>" if projected else "")
                ),
                hoverinfo="text", showlegend=False,
            ))
            x += 1
            
        # Add zone label on the left
        fig.add_annotation(
            x=-0.5, y=y, text=f"<b>{zone.upper()}</b>", showarrow=False, 
            font=dict(size=14, color="#334155"), xanchor="right"
        )
            
    fig.update_layout(
        height=120 * num_zones, xaxis=dict(visible=False), 
        yaxis=dict(visible=False, range=[-0.5, num_zones]),
        margin=dict(l=80, r=20, t=20, b=20),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
    )
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
    st.markdown(
        "<div style='font-size: 0.85em; color: #64748b; margin-top: -10px; margin-bottom: 20px; line-height: 1.8;'>"
        "<b>State:</b> <span class='badge-working'>working</span> &nbsp; "
        "<span class='badge-blocked'>blocked</span> &nbsp; "
        "<span class='badge-starved'>starved</span> &nbsp; "
        "<span class='badge-down'>down</span><br/>"
        "<b>Sensors:</b> ◆ Tier A (rich) &nbsp; ■ Tier B (partial) &nbsp; ● Tier C (barcode only)<br/>"
        "<b>Simulation:</b> <span style='border: 1.5px solid #ef4444; padding: 2px 4px; border-radius: 4px; color: #ef4444; font-weight: bold;'>⚠ RED OUTLINE</span> = projected to block/starve if unaddressed"
        "</div>",
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

    st.divider()

    # --- Active Detections as a dataframe table ---
    st.subheader("🔍 Active Detections")
    st.markdown("<div class='caption-text'>These are anomalies happening <b>right now</b> on the factory floor. Each row is a sensor reading that has crossed its normal operating range.</div>", unsafe_allow_html=True)
    import pandas as _pd
    if not state.detections:
        st.info("✅ No active detections — all signals within normal operating range.")
    else:
        det_rows = []
        for d in sorted(state.detections, key=lambda d: d.ts, reverse=True)[:30]:
            det_rows.append({
                "Time": d.ts.strftime("%H:%M:%S"),
                "Station": d.station_id,
                "Signal": d.signal,
                "Severity": d.severity.upper() if d.confirmed else "UNCONFIRMED",
                "Reading": round(d.value, 2),
                "Reason": d.reason,
                "Confirmed": "✅ Yes" if d.confirmed else "⏳ Pending",
            })
        st.dataframe(_pd.DataFrame(det_rows), use_container_width=True, hide_index=True, height=250)
        st.caption(f"{len(det_rows)} recent detection(s) shown.")

    st.divider()

    # --- Degradation Forecasts as a dataframe table ---
    st.subheader("📈 Degradation Forecasts")
    st.markdown("<div class='caption-text'>Unlike detections (which react to what's happening <i>now</i>), forecasts <b>predict the future</b>. The twin analyzes the drift trend of a sensor signal and tells you <b>how many hours remain</b> before it crosses its spec limit — giving you time to schedule maintenance before anything breaks down.</div>", unsafe_allow_html=True)
    active = {k: v for k, v in state.forecasts.items()
              if v.status in (STATUS_FORECASTABLE, STATUS_DRIFTING, STATUS_CROSSED)}
    if not active:
        st.success("✅ All monitored signals stable — no degradation trends detected.")
    else:
        fc_rows = []
        for (station_id, signal), fc in active.items():
            conf_level = "High" if fc.confidence >= 0.75 else ("Medium" if fc.confidence >= 0.45 else "Low")
            if fc.status == STATUS_FORECASTABLE:
                time_left = f"{fc.hours_to_limit:.1f}h (range {fc.hours_to_limit_ci[0]:.1f}–{fc.hours_to_limit_ci[1]:.1f}h)"
                status_label = "⚠️ Drifting"
            elif fc.status == STATUS_CROSSED:
                time_left = "🔴 Spec Limit Reached"
                status_label = "🔴 Crossed"
            else:
                time_left = "Monitoring..."
                status_label = fc.status.capitalize()
            fc_rows.append({
                "Station": station_id,
                "Signal": signal,
                "Status": status_label,
                "Time to Spec Limit": time_left,
                "Confidence": conf_level,
                "Samples": fc.n_samples,
                "Sensor Tier": state.tier_of.get(station_id, "?"),
            })
        st.dataframe(_pd.DataFrame(fc_rows), use_container_width=True, hide_index=True)

    active = {k: v for k, v in state.forecasts.items()
              if v.status in (STATUS_FORECASTABLE, STATUS_DRIFTING, STATUS_CROSSED)}
    st.divider()
    st.subheader("Recommendations")
    st.markdown("<div class='caption-text'>Actionable intelligence for operators. This twin never writes to the PLC directly.</div>", unsafe_allow_html=True)
    if "recommendations" not in st.session_state:
        st.session_state.recommendations = {}
        
    store = st.session_state.recommendations

    # Most recent CONFIRMED detection per (station, signal)
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
            fresh = forecast_recommendation(station_id, signal, fc, state.as_of)
        elif (station_id, signal) in active_detections:
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

    for stale_key in [k for k in store if k not in active_keys]:
        del store[stale_key]

    if not store:
        st.success("No recommendations pending.")
    for key, rec in list(store.items()):
        
        # Apply CSS class based on status
        css_class = f"rec-card-{rec.status}"
        
        # Render custom HTML for card
        st.markdown(
            f"""
            <div class="{css_class}" style="background-color: #ffffff; padding: 16px; border: 1px solid #e2e8f0; border-radius: 6px; margin-bottom: 12px; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
                    <div style="font-weight: 600; color: #1e293b; font-size: 1.05em;">{rec.title}</div>
                    <div>{STATUS_BADGE[rec.status]}</div>
                </div>
                <div style="color: #475569; font-size: 0.9em; line-height: 1.4;">{rec.detail}</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
        c1, _ = st.columns(2)
        if rec.status == STATUS_PROPOSED:
            if c1.button("Acknowledge", key=f"ack_{key}"):
                rec.advance()
                st.rerun()
        elif rec.status == STATUS_ACKNOWLEDGED:
            if c1.button("Mark executed by human", key=f"exec_{key}"):
                rec.advance()
                st.rerun()

    st.divider()
    st.subheader("At-risk vehicles")
    st.markdown("<div class='caption-text'>Latent defects inferred via genealogy — inspect these VINs before shipment.</div>", unsafe_allow_html=True)
    if state.at_risk.empty:
        st.info("No vehicles currently flagged at risk.")
    else:
        show = state.at_risk[[
            "vin", "variant", "origin_station", "last_known_station",
            "expected_surfacing_station", "eta", "risk_score", "confidence",
        ]]
        st.dataframe(show, use_container_width=True, hide_index=True)
        st.caption(f"{len(show)} vehicles carrying elevated risk downstream.")
