from __future__ import annotations

import numpy as np
import pandas as pd

from src.optimization import (
    strategic_base_weights,
    apply_tactical_overlays,
    enforce_cash_floor,
    volatility_target_scale,
    smooth_weights,
    equal_weight,
)
from src.risk import estimate_covariance


def build_expected_returns(
    returns_window: pd.DataFrame,
    current_tactical_signal: pd.Series,
) -> pd.Series:
    hist_3m = returns_window.tail(3).mean()
    hist_6m = returns_window.tail(6).mean()
    hist_12m = returns_window.mean()

    tactical = current_tactical_signal.reindex(returns_window.columns).fillna(0.0)
    tactical_rank = tactical.rank(pct=True) - 0.5

    expected = 0.20 * hist_3m + 0.35 * hist_6m + 0.45 * hist_12m + 0.05 * tactical_rank
    return expected.fillna(0.0)


def run_dynamic_backtest(
    portfolio_prices: pd.DataFrame,
    portfolio_returns: pd.DataFrame,
    tactical_signals: pd.DataFrame,
    regime_series: pd.Series,
    config,
) -> pd.DataFrame:
    lookback = config.lookback_months
    results = []
    prev_weights = None

    trend_ma = portfolio_prices.rolling(config.trend_window_months).mean()
    trend_flags_full = (portfolio_prices > trend_ma).astype(float).reindex(portfolio_returns.index)

    for i, dt in enumerate(portfolio_returns.index):
        if i < lookback:
            continue

        window_returns = portfolio_returns.iloc[i - lookback:i].dropna(axis=1)
        dt_assets = window_returns.columns.tolist()
        if len(dt_assets) < 3:
            continue

        regime = int(regime_series.loc[dt])
        tactical_signal = tactical_signals.loc[dt, dt_assets]
        trend_flags = trend_flags_full.loc[dt, dt_assets]

        cov = estimate_covariance(
            returns_window=window_returns[dt_assets],
            method=config.covariance_method,
            ewma_lambda=config.ewma_lambda,
        )
        expected_returns = build_expected_returns(window_returns[dt_assets], tactical_signal)

        base_weights = strategic_base_weights(regime, expected_returns, cov, config)
        tilted_weights = apply_tactical_overlays(base_weights, tactical_signal, trend_flags, regime, config)
        defensive_weights = enforce_cash_floor(tilted_weights, regime, config.cash_ticker, config)
        vol_targeted_weights = volatility_target_scale(
            defensive_weights,
            cov,
            config.annual_target_volatility,
            config.cash_ticker,
        )
        final_weights = smooth_weights(prev_weights, vol_targeted_weights, config.smoothing_alpha)

        final_weights = final_weights.reindex(portfolio_returns.columns).fillna(0.0)
        final_weights = final_weights / final_weights.sum()

        realized_return = float((final_weights * portfolio_returns.loc[dt]).sum())

        if prev_weights is None:
            turnover = np.nan
            cost = 0.0
        else:
            aligned_prev = prev_weights.reindex(final_weights.index).fillna(0.0)
            turnover = float(np.abs(final_weights - aligned_prev).sum())
            cost = turnover * (config.transaction_cost_bps / 10000.0)

        net_return = realized_return - cost

        row = {
            "date": dt,
            "regime": regime,
            "gross_return": realized_return,
            "transaction_cost": cost,
            "turnover": turnover,
            "portfolio_return": net_return,
        }
        for asset, weight in final_weights.items():
            row[f"w_{asset}"] = weight

        results.append(row)
        prev_weights = final_weights.copy()

    result_df = pd.DataFrame(results).set_index("date")
    return result_df


def run_static_benchmarks(
    portfolio_returns: pd.DataFrame,
    config,
) -> dict[str, pd.Series]:
    aligned = portfolio_returns.copy()

    ew = equal_weight(list(aligned.columns))
    eq_weight_returns = aligned.mul(ew, axis=1).sum(axis=1)

    # Static 60/40 proxy
    static_weights = pd.Series(0.0, index=aligned.columns)
    if config.benchmark_6040_equity_ticker in static_weights.index:
        static_weights[config.benchmark_6040_equity_ticker] = 0.60
    if config.benchmark_6040_bond_ticker in static_weights.index:
        static_weights[config.benchmark_6040_bond_ticker] = 0.40
    if static_weights.sum() == 0:
        static_weights = ew.copy()
    else:
        static_weights = static_weights / static_weights.sum()
    static_6040_returns = aligned.mul(static_weights, axis=1).sum(axis=1)

    inv_vol = 1.0 / aligned.std().replace(0, np.nan)
    inv_vol = inv_vol / inv_vol.sum()
    inv_vol_returns = aligned.mul(inv_vol.fillna(0.0), axis=1).sum(axis=1)

    spy_buy_hold = aligned[config.benchmark_6040_equity_ticker].copy() if config.benchmark_6040_equity_ticker in aligned.columns else eq_weight_returns.copy()

    return {
        "Equal Weight": eq_weight_returns,
        "Static 60/40 Proxy": static_6040_returns,
        "Inverse Volatility": inv_vol_returns,
        "SPY Buy & Hold": spy_buy_hold,
    }
