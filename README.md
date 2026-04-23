# Adaptive Portfolio Project V2
## Hybrid Regime + Momentum + Volatility-Targeting Portfolio Management System

This is **Version 2** of the adaptive multi-asset portfolio project. It is built as a separate project so you can run it independently from Version 1 and compare:

- cumulative growth,
- drawdowns,
- rolling Sharpe,
- regime classification,
- portfolio weights,
- yearly asset allocation,
- and final performance metrics.

## What changed from V1

Version 1 proved the framework. Version 2 is designed to be **more competitive and more realistic** by introducing:

1. a richer market-state model,
2. tactical momentum overlays,
3. trend filters,
4. volatility targeting,
5. smarter defensive behavior,
6. weight smoothing to reduce whipsawing,
7. stronger charting and reporting.

## Main idea

The strategy does not rely on one static allocation rule. It combines:

- **regime detection** from market stress signals,
- **cross-asset trend and momentum**,
- **risk-aware optimization**,
- **volatility targeting**,
- **cash defense during stress**,
- and **turnover-aware rebalancing**.

## Portfolio assets

- SPY - US Equities
- EFA - Developed ex-US Equities
- EEM - Emerging Markets
- TLT - Long-Term Bonds
- IEF - Intermediate Bonds
- GLD - Gold
- DBC - Commodities
- VNQ - Real Estate
- BIL - Cash / T-Bills

## Signal assets

These are used for market condition inference, not necessarily as portfolio holdings:
- ^VIX
- HYG
- LQD

## Core outputs

The project writes:
- metrics table,
- portfolio return history,
- benchmark history,
- regime summary,
- chart pack,
- yearly average weights.

## How to run

```bash
pip install -r requirements.txt
python run_project.py
```

## Main charts produced

Inside `outputs/charts/`:
- `01_growth_of_1.png`
- `02_drawdown_comparison.png`
- `03_rolling_sharpe.png`
- `04_regime_timeline.png`
- `05_portfolio_weights.png`
- `06_regime_colored_growth.png`
- `07_yearly_weight_labels.png`

## Why this version should improve

Version 1 could lag strong bull markets because it was often too conservative. Version 2 explicitly tries to solve that by:
- letting offensive assets carry more weight in calm-growth regimes,
- using momentum tilts,
- reducing exposure when an asset loses trend support,
- dynamically scaling risk to a target volatility,
- raising cash only when stress is truly elevated.

## Suggested CV bullet

Built a hybrid regime- and momentum-driven multi-asset portfolio allocation engine with volatility targeting, stress-aware cash overlays, turnover smoothing, and full benchmark backtesting, designed as an enhanced successor to an earlier adaptive portfolio framework.
