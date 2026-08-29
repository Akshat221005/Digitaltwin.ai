"""OPTIONAL Stage 2: a learned model layered on top of the day-one spec logic
(BUILD_SPEC.md §7.5). Predicts per-VIN downstream inspection failure from
genealogy features, once enough historical data exists to train on.

This resolves the tension between "no training data needed on day one"
(BUILD_SPEC.md §3.3, a hard constraint) and "historical logs eventually train a
model": the spec-limit/EWMA/forecast/inference stack in models/{detect,forecast,
infer}.py works with zero history and never depends on this module. This is a
Week-4+ enhancement, not a prerequisite — if it isn't trained, nothing else in
this codebase breaks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


@dataclass
class TrainedModel:
    model: HistGradientBoostingClassifier
    feature_names: list[str]
    feature_importances: pd.Series  # sorted descending, for "always show top contributions"
    test_auc: float
    n_train: int
    n_test: int


def build_vin_features(events_df: pd.DataFrame, station_configs: list) -> pd.DataFrame:
    """One row per VIN: dwell stats, tier-A/B signal deviations from nominal,
    and variant — the genealogy features BUILD_SPEC.md §7.5 asks for."""
    signal_specs = {
        (s.id, sig): spec for s in station_configs for sig, spec in (s.signals or {}).items()
    }
    signal_cols = sorted({sig for (_, sig) in signal_specs.keys()})

    rows = []
    for vin, g in events_df.groupby("vin"):
        row = {
            "vin": vin,
            "variant": g["variant"].iloc[0],
            "n_stations_visited": len(g),
            "mean_dwell_s": g["dwell_s"].mean(),
            "max_dwell_s": g["dwell_s"].max(),
            "dwell_std_s": g["dwell_s"].std() if len(g) > 1 else 0.0,
        }
        for sig in signal_cols:
            if sig not in g.columns:
                row[f"max_abs_dev_{sig}"] = 0.0
                continue
            vals = g[sig].dropna()
            if vals.empty:
                row[f"max_abs_dev_{sig}"] = 0.0
                continue
            # deviation as a fraction of that signal's spec span, so magnitudes
            # are comparable across signals with very different units
            devs = []
            for station_id, val in zip(g.loc[vals.index, "station_id"], vals):
                spec = signal_specs.get((station_id, sig))
                if spec is None:
                    continue
                span = spec["spec_high"] - spec["spec_low"]
                devs.append(abs(val - spec["nominal"]) / span if span else 0.0)
            row[f"max_abs_dev_{sig}"] = max(devs) if devs else 0.0
        rows.append(row)

    df = pd.DataFrame(rows)
    return pd.get_dummies(df, columns=["variant"], prefix="variant")


def train(features_df: pd.DataFrame, ground_truth_df: pd.DataFrame,
          random_state: int = 42) -> TrainedModel:
    """Trains on VINs with a resolved outcome (surfaced_at_station not null) —
    reworked=True is the positive class (a confirmed downstream failure)."""
    resolved = ground_truth_df[ground_truth_df.surfaced_at_station.notna()]
    label_by_vin = resolved.groupby("vin")["reworked"].max()  # any confirmed defect -> positive

    df = features_df[features_df.vin.isin(label_by_vin.index)].copy()
    df["label"] = df.vin.map(label_by_vin).astype(int)

    feature_cols = [c for c in df.columns if c not in ("vin", "label")]
    X, y = df[feature_cols], df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y if y.nunique() > 1 else None,
    )
    model = HistGradientBoostingClassifier(random_state=random_state, max_depth=4)
    model.fit(X_train, y_train)

    if y_test.nunique() > 1:
        from sklearn.metrics import roc_auc_score
        test_auc = float(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
    else:
        test_auc = float("nan")  # degenerate: not enough positive examples yet

    importances = permutation_importance(
        model, X_test, y_test, n_repeats=5, random_state=random_state, n_jobs=1,
    )
    importance_series = pd.Series(
        importances.importances_mean, index=feature_cols,
    ).sort_values(ascending=False)

    return TrainedModel(
        model=model, feature_names=feature_cols, feature_importances=importance_series,
        test_auc=test_auc, n_train=len(X_train), n_test=len(X_test),
    )


def score_with_contributions(trained: TrainedModel, features_row: pd.Series, top_k: int = 5) -> dict:
    """Score one VIN and return its probability alongside its top-k feature
    contributions — BUILD_SPEC.md §7.5: 'always show top feature contributions
    alongside any score.'"""
    x = features_row[trained.feature_names].to_frame().T
    proba = float(trained.model.predict_proba(x)[0, 1])
    top_features = trained.feature_importances.head(top_k)
    contributions = {
        feat: {"value": float(x[feat].iloc[0]), "global_importance": float(imp)}
        for feat, imp in top_features.items()
    }
    return {"probability": proba, "top_contributions": contributions}
