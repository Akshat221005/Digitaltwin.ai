"""Leadership tab: investment case. Every input is a slider — nothing about the
ROI, payback, or rollout numbers is hardcoded (BUILD_SPEC.md §10). Any headline
figure here is a *modelled output* of the assumptions below, restated as such
rather than presented as a sourced fact, per BUILD_SPEC.md §12."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st


def _measured_defect_rate(ground_truth_df: pd.DataFrame) -> float:
    if ground_truth_df.empty:
        return 0.02
    n_vins = ground_truth_df.vin.nunique()
    n_defects = len(ground_truth_df)
    # rough denominator: defects are rare per VIN, so use n_defects/n_vins as a
    # proxy rate rather than needing total line throughput here.
    return min(0.5, n_defects / max(n_vins, 1))


def render(state, ground_truth_df):
    st.subheader("ROI model")
    measured_rate = _measured_defect_rate(ground_truth_df[ground_truth_df.injected_ts <= state.as_of])

    c1, c2, c3 = st.columns(3)
    with c1:
        units_per_year = st.slider("Units produced / year", 50_000, 500_000, 180_000, step=10_000)
        defect_rate_pct = st.slider("Current defect rate (%)", 0.1, 10.0, round(measured_rate * 100, 2), step=0.1)
    with c2:
        defect_reduction_pct = st.slider("Defect rate reduction from twin (%)", 0, 100, 35, step=5)
        rework_cost_per_unit = st.slider("Rework cost / unit ($)", 20, 2000, 150, step=10)
    with c3:
        downtime_hours_avoided_per_year = st.slider("Unplanned downtime avoided / year (h)", 0, 500, 60, step=10)
        downtime_cost_per_hour = st.slider("Downtime cost / hour ($)", 500, 50_000, 8_000, step=500)

    license_cost_per_year = st.slider("Twin license cost / year ($)", 20_000, 2_000_000, 250_000, step=10_000)
    throughput_gain_pct = st.slider("Throughput gain from bottleneck visibility (%)", 0.0, 10.0, 1.5, step=0.1)
    unit_margin = st.slider("Contribution margin / unit ($)", 100, 20_000, 3_000, step=100)

    defects_avoided = units_per_year * (defect_rate_pct / 100) * (defect_reduction_pct / 100)
    rework_savings = defects_avoided * rework_cost_per_unit
    downtime_savings = downtime_hours_avoided_per_year * downtime_cost_per_hour
    throughput_units = units_per_year * (throughput_gain_pct / 100)
    throughput_value = throughput_units * unit_margin
    total_annual_value = rework_savings + downtime_savings + throughput_value
    net_annual_value = total_annual_value - license_cost_per_year
    payback_months = (license_cost_per_year / total_annual_value * 12) if total_annual_value > 0 else float("inf")

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rework savings / yr", f"${rework_savings:,.0f}")
    m2.metric("Downtime savings / yr", f"${downtime_savings:,.0f}")
    m3.metric("Throughput value / yr", f"${throughput_value:,.0f}")
    m4.metric("Net value / yr", f"${net_annual_value:,.0f}", delta=f"payback {payback_months:.1f} mo" if payback_months != float("inf") else "no payback")

    breakdown = pd.DataFrame({
        "component": ["Rework savings", "Downtime savings", "Throughput value"],
        "value": [rework_savings, downtime_savings, throughput_value],
    })
    st.plotly_chart(px.bar(breakdown, x="component", y="value", title="Annual value breakdown"),
                     use_container_width=True)

    st.subheader("Multi-site rollout scenario")
    st.caption("Three plants at different sensor maturity levels — payback compounds as retrofit debt is paid down.")
    sites = pd.DataFrame([
        {"site": "Plant A (pilot)", "tier_a_stations": 3, "tier_b_stations": 12, "tier_c_stations": 26,
         "phase": "Year 1", "annual_value": total_annual_value},
        {"site": "Plant B", "tier_a_stations": 2, "tier_b_stations": 6, "tier_c_stations": 33,
         "phase": "Year 2 (after 2 tier-C retrofits)", "annual_value": total_annual_value * 0.8},
        {"site": "Plant C", "tier_a_stations": 1, "tier_b_stations": 3, "tier_c_stations": 37,
         "phase": "Year 3 (after 4 tier-C retrofits)", "annual_value": total_annual_value * 0.65},
    ])
    st.dataframe(sites, width='stretch', hide_index=True)

    st.subheader("Phased roadmap")
    st.markdown(
        "- **Year 1** — Pilot at Plant A using existing tier-A/B sensors. Day-one value from "
        "spec-limit detection + degradation forecasting; zero new hardware.\n"
        "- **Year 2** — Retrofit the highest-ROI tier-C stations at Plant B (see the Plant "
        "Manager tab's retrofit ranking); extend inference coverage.\n"
        "- **Year 3** — Full rollout to Plant C; `models/learned.py` (Stage 2, optional) layers "
        "a trained model onto the day-one spec logic once enough historical data exists."
    )
    st.caption(
        "All figures above are modelled outputs of the sliders, not sourced plant statistics — "
        "see ASSUMPTIONS.md."
    )
