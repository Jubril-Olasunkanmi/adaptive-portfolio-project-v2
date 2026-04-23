from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from config import REGIME_NAME_MAP


def compute_stress_score(features: pd.DataFrame) -> pd.Series:
    score = (
        features["market_vol_6m"].rank(pct=True)
        + features["market_vol_12m"].rank(pct=True)
        + (-features["market_return_6m"]).rank(pct=True)
        + (-features["market_drawdown"]).rank(pct=True)
        + features["avg_pairwise_corr"].rank(pct=True)
        + (-features["trend_breadth"]).rank(pct=True)
        + features["vix_level"].rank(pct=True)
        + (-features["credit_ratio_trend"]).rank(pct=True)
    ) / 8.0
    return score


def deterministic_regime_labels(features: pd.DataFrame) -> pd.Series:
    stress = compute_stress_score(features)
    labels = pd.Series(index=features.index, dtype=int)

    labels[stress <= 0.30] = 0
    labels[(stress > 0.30) & (stress <= 0.55)] = 1
    labels[(stress > 0.55) & (stress <= 0.75)] = 2
    labels[stress > 0.75] = 3
    return labels.astype(int)


def cluster_overlay_regimes(features: pd.DataFrame, n_regimes: int, random_state: int) -> pd.Series:
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    gmm = GaussianMixture(
        n_components=n_regimes,
        covariance_type="full",
        random_state=random_state,
    )
    gmm_labels = gmm.fit_predict(X)

    km = KMeans(n_clusters=n_regimes, random_state=random_state, n_init=20)
    km_labels = km.fit_predict(X)

    temp = features.copy()
    temp["gmm"] = gmm_labels
    temp["km"] = km_labels

    gmm_feature_cols = [c for c in temp.columns if c != "gmm" and c != "km"]
    km_feature_cols = gmm_feature_cols

    gmm_scores = temp.groupby("gmm").apply(lambda df: compute_stress_score(df[gmm_feature_cols]).mean())
    km_scores = temp.groupby("km").apply(lambda df: compute_stress_score(df[km_feature_cols]).mean())

    gmm_map = {cluster: i for i, cluster in enumerate(gmm_scores.sort_values().index.tolist())}
    km_map = {cluster: i for i, cluster in enumerate(km_scores.sort_values().index.tolist())}

    ordered_gmm = pd.Series(gmm_labels, index=features.index).map(gmm_map).astype(int)
    ordered_km = pd.Series(km_labels, index=features.index).map(km_map).astype(int)

    combined = ((ordered_gmm + ordered_km) / 2).round().astype(int)
    combined = combined.clip(lower=0, upper=n_regimes - 1)
    return combined


def fit_predict_regimes(features: pd.DataFrame, config) -> pd.Series:
    det = deterministic_regime_labels(features)

    if config.use_cluster_overlay:
        overlay = cluster_overlay_regimes(features, config.n_regimes, config.random_state)
        combined = ((det + overlay) / 2).round().astype(int)
        combined = combined.clip(lower=0, upper=config.n_regimes - 1)
    else:
        combined = det

    combined.name = "regime"
    return combined


def summarize_regimes(features: pd.DataFrame, regime_series: pd.Series) -> pd.DataFrame:
    merged = features.join(regime_series, how="inner")
    summary = merged.groupby("regime").agg(["mean", "std", "count"])
    return summary


def add_regime_names(regime_series: pd.Series) -> pd.Series:
    return regime_series.map(REGIME_NAME_MAP)
