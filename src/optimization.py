from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


RISKY_ASSETS = {"SPY", "EFA", "EEM", "VNQ", "DBC"}
DEFENSIVE_ASSETS = {"TLT", "IEF", "GLD", "BIL"}


def _clean_covariance(cov: pd.DataFrame) -> np.ndarray:
    mat = cov.fillna(0.0).values
    mat = 0.5 * (mat + mat.T)
    mat += np.eye(mat.shape[0]) * 1e-8
    return mat


def _base_constraints(n_assets: int, min_weight: float, max_weight: float):
    bounds = [(min_weight, max_weight) for _ in range(n_assets)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    return bounds, constraints


def equal_weight(asset_names: list[str]) -> pd.Series:
    return pd.Series(np.ones(len(asset_names)) / len(asset_names), index=asset_names)


def min_variance_weights(cov: pd.DataFrame, min_weight: float, max_weight: float) -> pd.Series:
    sigma = _clean_covariance(cov)
    n = sigma.shape[0]
    x0 = np.ones(n) / n
    bounds, constraints = _base_constraints(n, min_weight, max_weight)

    def objective(w: np.ndarray) -> float:
        return float(w @ sigma @ w)

    res = minimize(objective, x0=x0, method="SLSQP", bounds=bounds, constraints=constraints)
    w = res.x if res.success else x0
    return pd.Series(w, index=cov.index)


def risk_parity_weights(cov: pd.DataFrame, min_weight: float, max_weight: float) -> pd.Series:
    sigma = _clean_covariance(cov)
    n = sigma.shape[0]
    x0 = np.ones(n) / n
    bounds, constraints = _base_constraints(n, min_weight, max_weight)

    def objective(w: np.ndarray) -> float:
        port_vol = np.sqrt(max(w @ sigma @ w, 1e-12))
        mrc = sigma @ w / port_vol
        trc = w * mrc
        target = np.ones(n) * port_vol / n
        return float(np.sum((trc - target) ** 2))

    res = minimize(objective, x0=x0, method="SLSQP", bounds=bounds, constraints=constraints)
    w = res.x if res.success else x0
    return pd.Series(w, index=cov.index)


def maximum_sharpe_proxy_weights(
    expected_returns: pd.Series,
    cov: pd.DataFrame,
    risk_aversion: float,
    min_weight: float,
    max_weight: float,
) -> pd.Series:
    sigma = _clean_covariance(cov)
    mu = expected_returns.fillna(0.0).values
    n = len(mu)
    x0 = np.ones(n) / n
    bounds, constraints = _base_constraints(n, min_weight, max_weight)

    def objective(w: np.ndarray) -> float:
        # Penalized mean-variance objective
        return float(-(w @ mu - 0.5 * risk_aversion * (w @ sigma @ w)))

    res = minimize(objective, x0=x0, method="SLSQP", bounds=bounds, constraints=constraints)
    w = res.x if res.success else x0
    return pd.Series(w, index=cov.index)


def strategic_base_weights(
    regime: int,
    expected_returns: pd.Series,
    cov: pd.DataFrame,
    config,
) -> pd.Series:
    asset_names = list(cov.index)

    if regime == 0:
        base = maximum_sharpe_proxy_weights(
            expected_returns=expected_returns,
            cov=cov,
            risk_aversion=2.5,
            min_weight=config.min_weight,
            max_weight=config.max_weight,
        )
    elif regime == 1:
        mv = maximum_sharpe_proxy_weights(
            expected_returns=expected_returns,
            cov=cov,
            risk_aversion=4.0,
            min_weight=config.min_weight,
            max_weight=config.max_weight,
        )
        rp = risk_parity_weights(cov, config.min_weight, config.max_weight)
        base = 0.55 * mv + 0.45 * rp
    elif regime == 2:
        mv = min_variance_weights(cov, config.min_weight, config.max_weight)
        rp = risk_parity_weights(cov, config.min_weight, config.max_weight)
        base = 0.50 * mv + 0.50 * rp
    else:
        base = min_variance_weights(cov, config.min_weight, config.max_weight)

    base = base / base.sum()
    return base.reindex(asset_names)


def apply_tactical_overlays(
    base_weights: pd.Series,
    tactical_signal: pd.Series,
    trend_flags: pd.Series,
    regime: int,
    config,
) -> pd.Series:
    w = base_weights.copy().astype(float)
    score = tactical_signal.reindex(w.index).fillna(0.0)
    trend = trend_flags.reindex(w.index).fillna(0.0)

    # Momentum tilt: positive score rewarded, negative score penalized
    score_std = score.std()
    normalized = score if (score_std == 0 or pd.isna(score_std)) else (score - score.mean()) / score_std

    tilt = 1.0 + config.momentum_strength * normalized.clip(-2, 2)
    w = w * tilt.clip(lower=0.10)

    # Trend filter: weak trend assets get partially de-risked
    below_trend = (trend < 1.0).astype(float)
    w = w * (1.0 - config.reversal_penalty_strength * below_trend)

    # Regime scaling: risky assets more aggressive in calm regime, toned down in stress
    risky_mask = w.index.isin(RISKY_ASSETS)
    defensive_mask = w.index.isin(DEFENSIVE_ASSETS)

    if regime == 0:
        w.loc[risky_mask] *= config.offensive_regime_multiplier
        w.loc[defensive_mask] *= 0.90
    elif regime == 2:
        w.loc[risky_mask] *= config.defensive_regime_multiplier
        w.loc[defensive_mask] *= 1.10
    elif regime == 3:
        w.loc[risky_mask] *= 0.45
        w.loc[defensive_mask] *= 1.30

    # Re-normalize
    w = w.clip(lower=0.0)
    if w.sum() <= 0:
        w = pd.Series(np.ones(len(w)) / len(w), index=w.index)
    else:
        w = w / w.sum()

    return w


def enforce_cash_floor(weights: pd.Series, regime: int, cash_ticker: str, config) -> pd.Series:
    w = weights.copy()
    if cash_ticker not in w.index:
        return w / w.sum()

    if regime <= 1:
        cash_floor = config.base_cash_floor
    elif regime == 2:
        cash_floor = config.stress_cash_floor
    else:
        cash_floor = config.extreme_stress_cash_floor

    current_cash = w.get(cash_ticker, 0.0)
    if current_cash >= cash_floor:
        return w / w.sum()

    deficit = cash_floor - current_cash
    risky_assets = [a for a in w.index if a != cash_ticker]
    risky_total = w[risky_assets].sum()

    if risky_total > 0:
        scale = max((risky_total - deficit) / risky_total, 0.0)
        w.loc[risky_assets] = w.loc[risky_assets] * scale
    w.loc[cash_ticker] = cash_floor

    if w.sum() <= 0:
        w.loc[:] = 1.0 / len(w)
    else:
        w = w / w.sum()

    return w


def volatility_target_scale(
    weights: pd.Series,
    cov: pd.DataFrame,
    target_annual_vol: float,
    cash_ticker: str,
) -> pd.Series:
    sigma = _clean_covariance(cov.loc[weights.index, weights.index])
    w = weights.values.copy()
    monthly_vol = np.sqrt(max(w @ sigma @ w, 1e-12))
    annual_vol = monthly_vol * np.sqrt(12.0)

    if annual_vol <= 0:
        return weights

    scale = min(target_annual_vol / annual_vol, 1.35)
    scaled = weights.copy()

    if scale >= 1.0:
        scaled = scaled / scaled.sum()
        return scaled

    # Move residual into cash when scaling down
    residual = 1.0 - scale
    scaled = scaled * scale
    if cash_ticker in scaled.index:
        scaled.loc[cash_ticker] = scaled.get(cash_ticker, 0.0) + residual
    else:
        scaled = scaled / scaled.sum()

    scaled = scaled / scaled.sum()
    return scaled


def smooth_weights(
    prev_weights: pd.Series | None,
    target_weights: pd.Series,
    alpha: float,
) -> pd.Series:
    if prev_weights is None:
        return target_weights / target_weights.sum()

    aligned_prev = prev_weights.reindex(target_weights.index).fillna(0.0)
    aligned_target = target_weights.reindex(target_weights.index).fillna(0.0)
    w = alpha * aligned_target + (1.0 - alpha) * aligned_prev
    if w.sum() <= 0:
        w = pd.Series(np.ones(len(w)) / len(w), index=w.index)
    else:
        w = w / w.sum()
    return w
