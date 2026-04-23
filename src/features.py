from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_drawdown(returns: pd.Series, window: int) -> pd.Series:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    rolling_peak = wealth.rolling(window, min_periods=1).max()
    drawdown = wealth / rolling_peak - 1.0
    return drawdown


def average_pairwise_correlation(returns_window: pd.DataFrame) -> float:
    corr = returns_window.corr()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    return upper.stack().mean()


def _zscore(series: pd.Series, window: int = 36) -> pd.Series:
    rolling_mean = series.rolling(window, min_periods=max(12, window // 3)).mean()
    rolling_std = series.rolling(window, min_periods=max(12, window // 3)).std()
    z = (series - rolling_mean) / rolling_std.replace(0, np.nan)
    return z


def build_regime_features(
    portfolio_returns: pd.DataFrame,
    portfolio_prices: pd.DataFrame,
    signal_returns: pd.DataFrame,
    signal_prices: pd.DataFrame,
    config,
) -> pd.DataFrame:
    features = pd.DataFrame(index=portfolio_returns.index)

    balanced_proxy = portfolio_returns.mean(axis=1)

    # Core return / volatility state variables
    features["market_return_3m"] = balanced_proxy.rolling(3).sum()
    features["market_return_6m"] = balanced_proxy.rolling(6).sum()
    features["market_vol_3m"] = balanced_proxy.rolling(3).std()
    features["market_vol_6m"] = balanced_proxy.rolling(6).std()
    features["market_vol_12m"] = balanced_proxy.rolling(12).std()
    features["market_drawdown"] = rolling_drawdown(balanced_proxy, window=12)

    # Cross-sectional and correlation stress
    features["cross_sectional_dispersion"] = portfolio_returns.std(axis=1).rolling(3).mean()

    avg_corr = []
    for i in range(len(portfolio_returns)):
        if i < 11:
            avg_corr.append(np.nan)
        else:
            win = portfolio_returns.iloc[i - 11:i + 1]
            avg_corr.append(average_pairwise_correlation(win))
    features["avg_pairwise_corr"] = avg_corr

    # Relative performance signals
    eq_cols = [c for c in portfolio_returns.columns if c in {"SPY", "EFA", "EEM", "VNQ"}]
    bond_cols = [c for c in portfolio_returns.columns if c in {"TLT", "IEF", "BIL"}]
    def_cols = [c for c in portfolio_returns.columns if c in {"TLT", "IEF", "GLD", "BIL"}]

    if eq_cols and bond_cols:
        eq_ret = portfolio_returns[eq_cols].mean(axis=1)
        bond_ret = portfolio_returns[bond_cols].mean(axis=1)
        features["equity_minus_bond_3m"] = (eq_ret - bond_ret).rolling(3).sum()
    else:
        features["equity_minus_bond_3m"] = np.nan

    if eq_cols and def_cols:
        eq_ret = portfolio_returns[eq_cols].mean(axis=1)
        def_ret = portfolio_returns[def_cols].mean(axis=1)
        features["risk_on_minus_defensive_6m"] = (eq_ret - def_ret).rolling(6).sum()
    else:
        features["risk_on_minus_defensive_6m"] = np.nan

    # Signal-only indicators
    if "^VIX" in signal_prices.columns:
        vix = signal_prices["^VIX"].copy()
        features["vix_level"] = _zscore(vix)
        features["vix_change_1m"] = vix.pct_change()
    else:
        features["vix_level"] = np.nan
        features["vix_change_1m"] = np.nan

    if {"HYG", "LQD"}.issubset(signal_prices.columns):
        credit_ratio = signal_prices["HYG"] / signal_prices["LQD"]
        features["credit_ratio_trend"] = credit_ratio.pct_change(3)
        features["credit_ratio_z"] = _zscore(credit_ratio)
    else:
        features["credit_ratio_trend"] = np.nan
        features["credit_ratio_z"] = np.nan

    # Trend breadth
    trend_window = config.trend_window_months
    trend_flags = []
    for dt in portfolio_prices.index:
        history = portfolio_prices.loc[:dt].tail(trend_window)
        if len(history) < trend_window:
            trend_flags.append(np.nan)
            continue
        latest = history.iloc[-1]
        ma = history.mean()
        trend_flags.append((latest > ma).mean())
    features["trend_breadth"] = trend_flags

    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    return features


def compute_tactical_signals(
    portfolio_prices: pd.DataFrame,
    portfolio_returns: pd.DataFrame,
    config,
) -> pd.DataFrame:
    signals = pd.DataFrame(index=portfolio_returns.index, columns=portfolio_returns.columns, dtype=float)

    ma = portfolio_prices.rolling(config.trend_window_months).mean()
    mom_short = portfolio_prices.pct_change(config.short_momentum_months)
    mom_long = portfolio_prices.pct_change(config.long_momentum_months)

    ma = ma.reindex(portfolio_returns.index)
    mom_short = mom_short.reindex(portfolio_returns.index)
    mom_long = mom_long.reindex(portfolio_returns.index)

    # Tactical score: positive momentum rewarded, negative trend penalized
    score = 0.4 * mom_short + 0.6 * mom_long
    trend_support = (portfolio_prices.reindex(portfolio_returns.index) > ma).astype(float)
    score = score + 0.10 * trend_support

    signals.loc[:, :] = score
    return signals
