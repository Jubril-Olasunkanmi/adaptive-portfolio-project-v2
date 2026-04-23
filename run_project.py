from __future__ import annotations

import os
import warnings

warnings.filterwarnings("ignore")

from config import CONFIG, REGIME_NAME_MAP
from src.data_loader import download_price_panel, align_portfolio_and_signal_data
from src.features import build_regime_features, compute_tactical_signals
from src.regime import fit_predict_regimes, summarize_regimes
from src.backtest import run_dynamic_backtest, run_static_benchmarks
from src.evaluation import (
    compute_metrics,
    save_results,
    generate_charts,
    generate_regime_colored_growth_chart,
    generate_yearly_weight_chart,
)


def main() -> None:
    os.makedirs(CONFIG.output_dir, exist_ok=True)
    charts_dir = os.path.join(CONFIG.output_dir, "charts")

    portfolio_prices = download_price_panel(CONFIG.tickers, CONFIG.start_date, CONFIG.end_date)
    signal_prices = download_price_panel(CONFIG.signal_tickers, CONFIG.start_date, CONFIG.end_date)

    (
        monthly_portfolio_prices,
        monthly_portfolio_returns,
        monthly_signal_prices,
        monthly_signal_returns,
    ) = align_portfolio_and_signal_data(
        portfolio_prices=portfolio_prices,
        signal_prices=signal_prices,
        freq=CONFIG.resample_freq,
    )

    features = build_regime_features(
        portfolio_returns=monthly_portfolio_returns,
        portfolio_prices=monthly_portfolio_prices,
        signal_returns=monthly_signal_returns,
        signal_prices=monthly_signal_prices,
        config=CONFIG,
    )
    regime_series = fit_predict_regimes(features, CONFIG)
    regime_summary = summarize_regimes(features, regime_series)

    common_index = monthly_portfolio_returns.index.intersection(features.index)
    monthly_portfolio_prices = monthly_portfolio_prices.loc[common_index]
    monthly_portfolio_returns = monthly_portfolio_returns.loc[common_index]
    regime_series = regime_series.loc[common_index]

    tactical_signals = compute_tactical_signals(
        portfolio_prices=monthly_portfolio_prices,
        portfolio_returns=monthly_portfolio_returns,
        config=CONFIG,
    ).loc[common_index]

    dynamic_results = run_dynamic_backtest(
        portfolio_prices=monthly_portfolio_prices,
        portfolio_returns=monthly_portfolio_returns,
        tactical_signals=tactical_signals,
        regime_series=regime_series,
        config=CONFIG,
    )

    benchmark_results = run_static_benchmarks(
        portfolio_returns=monthly_portfolio_returns.loc[dynamic_results.index],
        config=CONFIG,
    )

    dynamic_metrics = compute_metrics(
        returns=dynamic_results["portfolio_return"],
        risk_free_rate=CONFIG.annual_risk_free_rate,
        turnover=dynamic_results["turnover"],
    )

    benchmark_metrics = {
        name: compute_metrics(returns=series.loc[dynamic_results.index], risk_free_rate=CONFIG.annual_risk_free_rate)
        for name, series in benchmark_results.items()
    }

    save_results(
        dynamic_results=dynamic_results,
        benchmark_results=benchmark_results,
        dynamic_metrics=dynamic_metrics,
        benchmark_metrics=benchmark_metrics,
        regime_summary=regime_summary,
        output_dir=CONFIG.output_dir,
    )

    generate_charts(
        dynamic_results=dynamic_results,
        benchmark_results=benchmark_results,
        risk_free_rate=CONFIG.annual_risk_free_rate,
        output_dir=charts_dir,
    )
    generate_regime_colored_growth_chart(
        dynamic_results=dynamic_results,
        benchmark_results=benchmark_results,
        output_dir=charts_dir,
    )
    generate_yearly_weight_chart(
        dynamic_results=dynamic_results,
        output_dir=charts_dir,
    )

    print("Adaptive Portfolio Project V2 completed successfully.")
    print(f"Charts saved to: {charts_dir}")

    print("\nDynamic Portfolio V2 Metrics")
    for k, v in dynamic_metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    print("\nBenchmarks")
    for name, metrics in benchmark_metrics.items():
        print(f"\n{name}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    regime_counts = dynamic_results["regime"].value_counts().sort_index()
    print("\nObserved Regime Counts")
    for regime, count in regime_counts.items():
        print(f"  {regime} - {REGIME_NAME_MAP.get(regime, regime)}: {count}")


if __name__ == "__main__":
    main()
