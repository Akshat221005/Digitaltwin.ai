"""DigitalTwin.ai — Streamlit entry point. One model, three views.

Run: streamlit run app/main.py
"""
from __future__ import annotations

import sys
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator.line import load_line_config  # noqa: E402
from simulator.run import load_scenarios, build_scenario  # noqa: E402
from twin.state import DEFAULT_LOOKBACK_HOURS, build_twin_state  # noqa: E402
from app import view_leadership, view_manager, view_supervisor  # noqa: E402

st.set_page_config(page_title="DigitalTwin.ai", layout="wide", initial_sidebar_state="expanded")

SCENARIOS = ["weld_drift_demo", "nominal", "noisy_line"]


@st.cache_resource
def get_line_config():
    return load_line_config("config/line.yaml")


@st.cache_data
def load_scenario_data(scenario: str):
    base = Path("data") / scenario
    events = pd.read_parquet(base / "events.parquet")
    status = pd.read_parquet(base / "status_intervals.parquet")
    ground_truth = pd.read_parquet(base / "ground_truth.parquet")
    degradation_truth = pd.read_parquet(base / "degradation_truth.parquet")
    return events, status, ground_truth, degradation_truth


@st.cache_data(show_spinner="Replaying line state...")
def compute_twin_state(scenario: str, as_of, lookback_hours: float):
    _cfg = get_line_config()
    events, status, _, _ = load_scenario_data(scenario)
    return build_twin_state(_cfg, events, status, as_of, lookback_hours=lookback_hours)


@st.cache_resource(show_spinner="Forking the line to project consequences...")
def build_live_sim_upto(scenario: str, as_of_iso: str):
    """Reconstructs a live LineSimulator fork at exactly `as_of` — the object
    models/cascade.py needs to fork further and project consequences forward.
    Only called when actually needed (a signal has crossed spec), since it
    means genuinely re-running the tick loop, not just reading parquet."""
    from datetime import datetime

    cfg = get_line_config()
    scenarios = load_scenarios("config/scenarios.yaml")
    scenario_cfg = scenarios[scenario]
    start_ts = datetime.fromisoformat(scenario_cfg["start_ts"])
    as_of_ts = datetime.fromisoformat(as_of_iso)
    run_duration_s = max(1, int((as_of_ts - start_ts).total_seconds()))
    sim, engine, _, _ = build_scenario(cfg, scenario_cfg, run_duration_s=run_duration_s)
    return sim


def main():
    with open("app/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    st.title("DigitalTwin.ai")
    st.markdown(
        "<p style='color: #64748b; font-size: 1.1em; margin-top: -10px; margin-bottom: 30px;'>"
        "Industrial Twin: Anomaly Detection, Degradation Forecasting & Quality Inference"
        "</p>",
        unsafe_allow_html=True
    )

    with st.sidebar:
        st.header("Control Panel")
        scenario = st.selectbox("Active Scenario", SCENARIOS, index=0, help="Select the dataset to analyze.")
        events, status, ground_truth, degradation_truth = load_scenario_data(scenario)

        min_ts = events.entry_ts.min().to_pydatetime()
        max_ts = events.exit_ts.max().to_pydatetime()
        default_ts = (min_ts + (max_ts - min_ts) * 0.55)

        if st.session_state.get("_scenario") != scenario:
            st.session_state._scenario = scenario
            st.session_state.as_of_ts = default_ts
            st.session_state.playing = False
        st.session_state.setdefault("as_of_ts", default_ts)
        st.session_state.setdefault("playing", False)

        st.subheader("⏱ Time Playback")
        play_col, reset_col = st.columns(2)
        if play_col.button("⏸ Pause" if st.session_state.playing else "▶ Play", use_container_width=True):
            st.session_state.playing = not st.session_state.playing
            st.rerun()
        if reset_col.button("⏮ Reset", use_container_width=True):
            st.session_state.as_of_ts = default_ts
            st.session_state.playing = False
            st.rerun()

        step_minutes = st.select_slider(
            "Playback speed (min/tick)", options=[1, 2, 5, 10, 15, 30], value=2,
        )

        if st.session_state.playing:
            next_ts = st.session_state.as_of_ts + timedelta(minutes=step_minutes)
            if next_ts >= max_ts:
                next_ts = max_ts
                st.session_state.playing = False
            st.session_state.as_of_ts = next_ts

        as_of = st.slider(
            "Historical 'As of'", min_value=min_ts, max_value=max_ts,
            value=st.session_state.as_of_ts, step=timedelta(minutes=1),
            format="MM/DD HH:mm",
        )
        st.session_state.as_of_ts = as_of

        st.caption(f"**Current Time: {as_of.strftime('%Y-%m-%d %H:%M:%S')}**")

        st.divider()
        st.caption(
            "🔒 **Read-Only Mode**: Twin acts as a passive monitor. "
            "It never writes back to the line PLC directly."
        )

    lookback_hours = DEFAULT_LOOKBACK_HOURS if scenario != "weld_drift_demo" else 12.0
    state = compute_twin_state(scenario, pd.Timestamp(as_of), lookback_hours)
    cfg = get_line_config()

    tab_sup, tab_mgr, tab_lead = st.tabs([
        "🛠 Supervisor (Real-time)", "📊 Plant Manager (Planning)", "💼 ROI (Leadership)"
    ])
    get_live_sim = lambda: build_live_sim_upto(scenario, as_of.isoformat())  # noqa: E731

    with tab_sup:
        view_supervisor.render(state, cfg, get_live_sim)
    with tab_mgr:
        view_manager.render(state, cfg, status, ground_truth)
    with tab_lead:
        view_leadership.render(state, ground_truth)

    if st.session_state.get("playing"):
        time.sleep(0.6)
        st.rerun()


if __name__ == "__main__":
    main()
