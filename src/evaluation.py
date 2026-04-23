from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import ASSET_NAME_MAP, REGIME_NAME_MAP


def annualized_return(returns: pd.Series, periods_per_year: int = 12) -> float:
    r = returns.dropna()
    if r.empty:
        return np.nan
    compounded = (1.0 + r).prod()
    n = len(r)
    return compounded ** (periods_per_year / n) - 1.0


def annualized_volatility(returns: pd.Series, periods_per_year: int = 12) -> float:
    return returns.dropna().std() * np.sqrt(periods_per_year)


def sharpe_ratio(returns: pd.Series, risk_free_rate: float, periods_per_year: int = 12) -> float:
    rf_per_period = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess = returns.dropna() - rf_per_period
    denom = excess.std()
    if denom == 0 or np.isnan(denom):
        return np.nan
    return excess.mean() / denom * np.sqrt(periods_per_year)


def sortino_ratio(returns: pd.Series, risk_free_rate: float, periods_per_year: int = 12) -> float:
    rf_per_period = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess = returns.dropna() - rf_per_period
    downside = excess[excess < 0]
    downside_std = downside.std()
    if downside_std == 0 or np.isnan(downside_std):
        return np.nan
    return excess.mean() / downside_std * np.sqrt(periods_per_year)


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns.dropna()).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    return dd.min()


def calmar_ratio(returns: pd.Series, periods_per_year: int = 12) -> float:
    cagr = annualized_return(returns, periods_per_year)
    mdd = abs(max_drawdown(returns))
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    return cagr / mdd


def value_at_risk(returns: pd.Series, alpha: float = 0.05) -> float:
    return returns.dropna().quantile(alpha)


def conditional_value_at_risk(returns: pd.Series, alpha: float = 0.05) -> float:
    r = returns.dropna()
    var = r.quantile(alpha)
    tail = r[r <= var]
    if tail.empty:
        return np.nan
    return tail.mean()


def hit_rate(returns: pd.Series) -> float:
    r = returns.dropna()
    if r.empty:
        return np.nan
    return (r > 0).mean()


def rolling_sharpe(returns: pd.Series, risk_free_rate: float, window: int = 12, periods_per_year: int = 12) -> pd.Series:
    rf_per_period = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess = returns - rf_per_period
    roll_mean = excess.rolling(window).mean()
    roll_std = excess.rolling(window).std()
    return (roll_mean / roll_std) * np.sqrt(periods_per_year)


def compute_drawdown_series(returns: pd.Series) -> pd.Series:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    peak = wealth.cummax()
    return wealth / peak - 1.0


def compute_metrics(
    returns: pd.Series,
    risk_free_rate: float,
    turnover: pd.Series | None = None,
) -> dict:
    metrics = {
        "CAGR": annualized_return(returns),
        "Annualized Volatility": annualized_volatility(returns),
        "Sharpe Ratio": sharpe_ratio(returns, risk_free_rate),
        "Sortino Ratio": sortino_ratio(returns, risk_free_rate),
        "Max Drawdown": max_drawdown(returns),
        "Calmar Ratio": calmar_ratio(returns),
        "VaR 5%": value_at_risk(returns),
        "CVaR 5%": conditional_value_at_risk(returns),
        "Hit Rate": hit_rate(returns),
    }
    if turnover is not None:
        metrics["Average Turnover"] = turnover.dropna().mean()
    return metrics


def save_results(
    dynamic_results: pd.DataFrame,
    benchmark_results: dict[str, pd.Series],
    dynamic_metrics: dict,
    benchmark_metrics: dict[str, dict],
    regime_summary: pd.DataFrame,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dynamic_results.to_csv(os.path.join(output_dir, "portfolio_results.csv"))

    benchmark_df = pd.DataFrame(benchmark_results)
    benchmark_df.to_csv(os.path.join(output_dir, "benchmark_results.csv"))

    metrics_df = pd.DataFrame({"Dynamic Strategy V2": dynamic_metrics})
    for name, metrics in benchmark_metrics.items():
        metrics_df[name] = pd.Series(metrics)
    metrics_df.to_csv(os.path.join(output_dir, "metrics.csv"))

    regime_summary.to_csv(os.path.join(output_dir, "regime_summary.csv"))


def generate_charts(
    dynamic_results: pd.DataFrame,
    benchmark_results: dict[str, pd.Series],
    risk_free_rate: float,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    dynamic_returns = dynamic_results["portfolio_return"].copy()
    benchmark_df = pd.DataFrame(benchmark_results).copy()

    common_index = dynamic_returns.index
    if not benchmark_df.empty:
        common_index = common_index.intersection(benchmark_df.index)

    dynamic_returns = dynamic_returns.loc[common_index]
    benchmark_df = benchmark_df.loc[common_index]

    combined_returns = pd.concat([dynamic_returns.rename("Dynamic Portfolio V2"), benchmark_df], axis=1).dropna(how="all")

    growth = (1.0 + combined_returns.fillna(0.0)).cumprod()
    plt.figure(figsize=(12, 6))
    for col in growth.columns:
        plt.plot(growth.index, growth[col], label=col, linewidth=2 if "Dynamic" in col else 1.6)
    plt.title("Growth of $1: Dynamic Portfolio V2 vs Benchmarks")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_growth_of_1.png"), dpi=300)
    plt.close()

    drawdown_df = pd.DataFrame({col: compute_drawdown_series(combined_returns[col]) for col in combined_returns.columns})
    plt.figure(figsize=(12, 6))
    for col in drawdown_df.columns:
        plt.plot(drawdown_df.index, drawdown_df[col], label=col, linewidth=2 if "Dynamic" in col else 1.6)
    plt.title("Drawdown Comparison")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_drawdown_comparison.png"), dpi=300)
    plt.close()

    rolling_sharpe_df = pd.DataFrame({
        col: rolling_sharpe(combined_returns[col], risk_free_rate=risk_free_rate, window=12)
        for col in combined_returns.columns
    })
    plt.figure(figsize=(12, 6))
    for col in rolling_sharpe_df.columns:
        plt.plot(rolling_sharpe_df.index, rolling_sharpe_df[col], label=col, linewidth=2 if "Dynamic" in col else 1.6)
    plt.axhline(0, linestyle="--")
    plt.title("12-Month Rolling Sharpe Ratio")
    plt.xlabel("Date")
    plt.ylabel("Rolling Sharpe")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_rolling_sharpe.png"), dpi=300)
    plt.close()

    regime_series = dynamic_results["regime"].copy()
    plt.figure(figsize=(12, 3))
    plt.step(regime_series.index, regime_series.values, where="post")
    plt.title("Market Regime Timeline")
    plt.xlabel("Date")
    plt.ylabel("Regime")
    plt.yticks(list(REGIME_NAME_MAP.keys()), [REGIME_NAME_MAP[k] for k in REGIME_NAME_MAP.keys()])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_regime_timeline.png"), dpi=300)
    plt.close()

    weight_cols = [c for c in dynamic_results.columns if c.startswith("w_")]
    weights_df = dynamic_results[weight_cols].copy()
    weights_df.columns = [ASSET_NAME_MAP.get(c.replace("w_", ""), c.replace("w_", "")) for c in weight_cols]

    plt.figure(figsize=(12, 7))
    plt.stackplot(weights_df.index, [weights_df[c] for c in weights_df.columns], labels=weights_df.columns)
    plt.title("Dynamic Portfolio Weights Through Time")
    plt.xlabel("Date")
    plt.ylabel("Weight")
    plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "05_portfolio_weights.png"), dpi=300)
    plt.close()


def generate_regime_colored_growth_chart(
    dynamic_results: pd.DataFrame,
    benchmark_results: dict[str, pd.Series],
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    dynamic_returns = dynamic_results["portfolio_return"].copy().dropna()
    regime_series = dynamic_results["regime"].copy().reindex(dynamic_returns.index).ffill()
    growth = (1.0 + dynamic_returns).cumprod()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(growth.index, growth.values, linewidth=2.3, label="Dynamic Portfolio V2")

    colors = {
        0: "#cfe8cf",
        1: "#f6efc1",
        2: "#f7d9b8",
        3: "#f2c6c6",
    }

    regimes = regime_series.values
    start = 0
    for i in range(1, len(regimes)):
        if regimes[i] != regimes[i - 1]:
            ax.axvspan(
                growth.index[start],
                growth.index[i - 1],
                color=colors.get(regimes[i - 1], "#eeeeee"),
                alpha=0.35,
            )
            start = i

    if len(regimes) > 0:
        ax.axvspan(
            growth.index[start],
            growth.index[-1],
            color=colors.get(regimes[-1], "#eeeeee"),
            alpha=0.35,
        )

    bench = pd.DataFrame(benchmark_results).reindex(dynamic_returns.index)
    for col in bench.columns:
        benchmark_growth = (1.0 + bench[col].fillna(0.0)).cumprod()
        ax.plot(benchmark_growth.index, benchmark_growth.values, linestyle="--", label=col, alpha=0.95)

    ax.set_title("Growth of $1 with Regime Background Shading")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "06_regime_colored_growth.png"), dpi=300)
    plt.close()


def generate_yearly_weight_chart(
    dynamic_results: pd.DataFrame,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    weight_cols = [c for c in dynamic_results.columns if c.startswith("w_")]
    if not weight_cols:
        return

    weights_df = dynamic_results[weight_cols].copy()
    weights_df.columns = [ASSET_NAME_MAP.get(c.replace("w_", ""), c.replace("w_", "")) for c in weight_cols]
    yearly_weights = weights_df.resample("YE").mean()
    yearly_weights.index = yearly_weights.index.year.astype(str)
    yearly_weights.to_csv(os.path.join(output_dir, "yearly_average_weights.csv"))

    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"
    ]

    fig, ax = plt.subplots(figsize=(14, 7))
    bottom = np.zeros(len(yearly_weights))

    for i, col in enumerate(yearly_weights.columns):
        values = yearly_weights[col].values
        ax.bar(
            yearly_weights.index,
            values,
            bottom=bottom,
            label=col,
            color=colors[i % len(colors)],
            edgecolor="white",
        )
        for j, v in enumerate(values):
            if v > 0.04:
                ax.text(
                    j,
                    bottom[j] + v / 2,
                    f"{v:.0%}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color="white",
                )
        bottom += values

    ax.set_title("Average Portfolio Allocation by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Weight (%)")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "07_yearly_weight_labels.png"), dpi=300)
    plt.close()
