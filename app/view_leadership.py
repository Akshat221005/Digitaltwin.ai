"""Leadership tab: ROI simulator — no emojis."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st


def _measured_defect_rate(ground_truth_df: pd.DataFrame) -> float:
    if ground_truth_df.empty:
        return 0.02
    n_vins = ground_truth_df.vin.nunique()
    n_defects = len(ground_truth_df)
    return min(0.5, n_defects / max(n_vins, 1))


def render(state, ground_truth_df):
    measured_rate = _measured_defect_rate(ground_truth_df[ground_truth_df.injected_ts <= state.as_of])

    st.markdown(
        "<div class='insight-box'>"
        "This is an interactive simulator. Use the sliders below to model your factory's numbers "
        "and instantly see how much value the digital twin generates. All calculations update in real time."
        "</div>",
        unsafe_allow_html=True
    )

    # Step 1
    st.markdown("---")
    st.markdown("### Step 1 — Tell Us About Your Factory")
    st.markdown(
        "<div class='caption-text'>Enter your factory's current baseline numbers. "
        "These are the costs you're <b>already incurring</b> before the digital twin.</div>",
        unsafe_allow_html=True
    )
    c1, c2 = st.columns(2)
    units_per_year = c1.slider(
        "Units produced per year",
        50_000, 500_000, 180_000, step=10_000,
        help="Total vehicles or units your factory assembles in a year."
    )
    defect_rate_pct = c2.slider(
        "Current defect rate (%)",
        0.1, 10.0, round(measured_rate * 100, 2), step=0.1,
        help="What percentage of finished units have a defect that needs reworking?"
    )
    rework_cost_per_unit = c1.slider(
        "Rework cost per defective unit ($)",
        20, 2000, 150, step=10,
        help="How much does it cost to fix one defective unit at end-of-line inspection?"
    )
    downtime_cost_per_hour = c2.slider(
        "Unplanned downtime cost per hour ($)",
        500, 50_000, 8_000, step=500,
        help="When the line unexpectedly stops, what does one hour of downtime cost?"
    )
    unit_margin = c1.slider(
        "Profit margin per unit ($)",
        100, 20_000, 3_000, step=100,
        help="How much profit does one additional unit sold generate?"
    )

    # Step 2
    st.markdown("---")
    st.markdown("### Step 2 — What Will the Digital Twin Improve?")
    st.markdown(
        "<div class='caption-text'>These sliders represent the <b>impact you expect</b> from deploying the digital twin.</div>",
        unsafe_allow_html=True
    )
    c3, c4 = st.columns(2)
    defect_reduction_pct = c3.slider(
        "Defect reduction from twin (%)",
        0, 100, 35, step=5,
        help="By catching drifts early, how much do you expect your defect rate to fall?"
    )
    downtime_hours_avoided_per_year = c4.slider(
        "Unplanned stoppages avoided per year (hours)",
        0, 500, 60, step=10,
        help="Because the twin forecasts breakdowns before they happen, how many hours of unplanned downtime can you avoid?"
    )
    throughput_gain_pct = st.slider(
        "Throughput gain from bottleneck visibility (%)",
        0.0, 10.0, 1.5, step=0.1,
        help="By knowing which station is slowing the line, you can re-balance work and produce more."
    )

    # Step 3
    st.markdown("---")
    st.markdown("### Step 3 — The Investment")
    st.markdown(
        "<div class='caption-text'>This is what you <b>pay</b> to run the digital twin.</div>",
        unsafe_allow_html=True
    )
    license_cost_per_year = st.slider(
        "Digital Twin license cost per year ($)",
        20_000, 2_000_000, 250_000, step=10_000,
        help="Annual software and operational cost of the digital twin platform."
    )

    # Calculations
    defects_avoided = units_per_year * (defect_rate_pct / 100) * (defect_reduction_pct / 100)
    rework_savings = defects_avoided * rework_cost_per_unit
    downtime_savings = downtime_hours_avoided_per_year * downtime_cost_per_hour
    throughput_units = units_per_year * (throughput_gain_pct / 100)
    throughput_value = throughput_units * unit_margin
    total_annual_value = rework_savings + downtime_savings + throughput_value
    net_annual_value = total_annual_value - license_cost_per_year
    payback_months = (license_cost_per_year / total_annual_value * 12) if total_annual_value > 0 else float("inf")

    # Results
    st.markdown("---")
    st.markdown("### Results: Where the Money Goes")
    st.markdown(
        "<div class='caption-text'>Based on your inputs above, here is the full financial picture broken down by category.</div>",
        unsafe_allow_html=True
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rework Savings / yr", f"${rework_savings:,.0f}")
    m2.metric("Downtime Savings / yr", f"${downtime_savings:,.0f}")
    m3.metric("Throughput Value / yr", f"${throughput_value:,.0f}")
    m4.metric(
        "Net Value / yr",
        f"${net_annual_value:,.0f}",
        delta=f"Payback in {payback_months:.1f} months" if payback_months != float("inf") else "No payback yet",
    )

    breakdown = pd.DataFrame({
        "Category": ["Rework Savings", "Downtime Savings", "Throughput Value", "Twin License Cost"],
        "Amount ($)": [rework_savings, downtime_savings, throughput_value, -license_cost_per_year],
    })
    fig = px.bar(
        breakdown, x="Category", y="Amount ($)",
        title="Annual Financial Breakdown",
        color="Category",
        color_discrete_sequence=["#22c55e", "#3b82f6", "#a855f7", "#ef4444"],
    )
    fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)

    if net_annual_value > 0:
        st.markdown(
            f"<div class='insight-box'><b>ROI Summary:</b> The digital twin generates <b>${total_annual_value:,.0f}/yr</b> in value "
            f"against a <b>${license_cost_per_year:,.0f}/yr</b> investment. "
            f"You break even in <b>{payback_months:.1f} months</b>.</div>",
            unsafe_allow_html=True
        )
    else:
        st.warning("Current assumptions show a net negative return. Try increasing the impact estimates or reducing the license cost.")

    # Rollout
    st.markdown("---")
    st.markdown("### Multi-Plant Rollout Plan")
    st.markdown(
        "<div class='caption-text'>"
        "The digital twin can scale from a pilot to a full multi-site deployment. "
        "Each year adds more coverage as dark stations get instrumented."
        "</div>",
        unsafe_allow_html=True
    )
    sites = pd.DataFrame([
        {"Phase": "Year 1 — Pilot", "Site": "Plant A", "Coverage": "Existing Tier A/B sensors only",
         "Projected Annual Value": f"${total_annual_value:,.0f}", "Notes": "Zero new hardware. Full forecasting and detection from Day 1."},
        {"Phase": "Year 2 — Expand", "Site": "Plant B", "Coverage": "Pilot + 2 Tier-C retrofits",
         "Projected Annual Value": f"${total_annual_value * 0.8:,.0f}", "Notes": "Instrument highest-ROI dark stations from retrofit ranking."},
        {"Phase": "Year 3 — Full Rollout", "Site": "Plant C", "Coverage": "Full coverage",
         "Projected Annual Value": f"${total_annual_value * 0.65:,.0f}", "Notes": "Learned model activates once enough historical data is available."},
    ])
    st.dataframe(sites, use_container_width=True, hide_index=True)

    st.markdown(
        "<div style='font-size: 0.85em; color: #64748b; margin-top: 15px;'>"
        "<b>Note:</b> All values are modelled estimates based on the sliders above."
        "</div>",
        unsafe_allow_html=True
    )
