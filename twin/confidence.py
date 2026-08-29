"""Confidence scoring (BUILD_SPEC.md §8) and retrofit ROI ranking (§8.1).

Every detection, forecast, and inference must carry a 0-1 confidence score.
Being visibly honest about uncertainty is the mechanism that protects floor
trust — never show a prediction without it.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

TIER_CONFIDENCE = {"A": 1.0, "B": 0.6, "C": 0.3}

LEVEL_HIGH = "High"
LEVEL_MEDIUM = "Medium"
LEVEL_LOW = "Low"


@dataclass
class Confidence:
    value: float           # 0..1
    level: str              # High / Medium / Low
    components: dict         # breakdown, for the hover tooltip


def sample_sufficiency(n_samples: int, required: int) -> float:
    return max(0.0, min(1.0, n_samples / required)) if required > 0 else 1.0


def model_agreement(signals_agree: list[bool]) -> float:
    """Fraction of independent signals (e.g. spec check, control chart, forecast)
    that agree something is wrong. Empty list -> neutral 0.5 (no corroboration
    available, not treated as disagreement)."""
    if not signals_agree:
        return 0.5
    return sum(1 for s in signals_agree if s) / len(signals_agree)


def historical_accuracy(station_id: str, accuracy_by_station: dict) -> float:
    """Fraction of past predictions at this station that were later confirmed
    correct. Defaults to a neutral 0.5 when no history exists yet — day-one
    value must not require historical data (BUILD_SPEC.md §2.3)."""
    return accuracy_by_station.get(station_id, 0.5)


def compute_confidence(sensor_tier: str, n_samples: int, required_samples: int,
                        signals_agree: list[bool], station_id: str,
                        accuracy_by_station: dict | None = None,
                        weights: dict | None = None) -> Confidence:
    weights = weights or {"tier": 0.4, "sufficiency": 0.25, "agreement": 0.20, "history": 0.15}
    accuracy_by_station = accuracy_by_station or {}

    tier_c = TIER_CONFIDENCE.get(sensor_tier, 0.3)
    suff_c = sample_sufficiency(n_samples, required_samples)
    agree_c = model_agreement(signals_agree)
    hist_c = historical_accuracy(station_id, accuracy_by_station)

    value = (
        weights["tier"] * tier_c + weights["sufficiency"] * suff_c
        + weights["agreement"] * agree_c + weights["history"] * hist_c
    )
    value = max(0.0, min(1.0, value))
    level = LEVEL_HIGH if value >= 0.75 else (LEVEL_MEDIUM if value >= 0.45 else LEVEL_LOW)
    return Confidence(
        value=value, level=level,
        components={"sensor_tier": tier_c, "sample_sufficiency": suff_c,
                    "model_agreement": agree_c, "historical_accuracy": hist_c},
    )


DEFAULT_REWORK_COST_PER_UNIT = 150.0  # see ASSUMPTIONS.md


def default_retrofit_cost_by_station(station_configs: list) -> dict:
    """$10k-$25k per tier-C station, varied deterministically by station index —
    an invented placeholder (see ASSUMPTIONS.md), not a sourced quote."""
    tier_c = [s.id for s in station_configs if s.sensor_tier == "C"]
    return {sid: 10_000 + (i % 4) * 5_000 for i, sid in enumerate(tier_c)}


# ---------------------------------------------------------------------------
# §8.1 Retrofit ROI ranking
# ---------------------------------------------------------------------------

def retrofit_ranking(defects_df: pd.DataFrame, station_configs: list, rework_cost_per_unit: float,
                      retrofit_cost_by_station: dict) -> pd.DataFrame:
    """Ranks tier-C (dark) stations by expected confidence gain per retrofit
    dollar: defects back-attributed to that station (via root_cause), weighted
    by rework cost, divided by estimated retrofit cost.

    root_cause for tier-C-origin defects is 'unspecified' / '<station>:background'
    (see simulator/defects.py) since there's no signal to attribute a specific
    deviation to — that's the whole point of instrumenting them.
    """
    tier_c_ids = {s.id for s in station_configs if s.sensor_tier == "C"}
    if defects_df.empty:
        return pd.DataFrame(columns=["station_id", "defect_count", "weighted_cost",
                                      "retrofit_cost", "payback_score"])

    dark_defects = defects_df[defects_df.injected_at_station.isin(tier_c_ids)]
    counts = dark_defects.groupby("injected_at_station").size().rename("defect_count")
    rows = []
    for station_id in tier_c_ids:
        n = int(counts.get(station_id, 0))
        weighted_cost = n * rework_cost_per_unit
        retrofit_cost = retrofit_cost_by_station.get(station_id, float("nan"))
        payback_score = weighted_cost / retrofit_cost if retrofit_cost else 0.0
        rows.append({
            "station_id": station_id, "defect_count": n, "weighted_cost": weighted_cost,
            "retrofit_cost": retrofit_cost, "payback_score": payback_score,
        })
    return pd.DataFrame(rows).sort_values("payback_score", ascending=False)
