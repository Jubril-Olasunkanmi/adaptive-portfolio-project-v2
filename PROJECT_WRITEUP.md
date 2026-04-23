# Adaptive Portfolio Project V2
## Full Thesis-Style Project Narrative

## 1. Introduction

Portfolio management in practice is not a static optimization problem. Markets rotate through calm expansions, volatile uptrends, policy shocks, inflation scares, credit stress, and crisis conditions. A single fixed allocation is rarely optimal across all such states. This project develops a more realistic portfolio framework that combines market regime classification with tactical momentum, trend filtering, and volatility targeting.

This Version 2 project is intentionally designed as a stronger successor to Version 1. While Version 1 demonstrated that an adaptive framework could be implemented end to end, Version 2 focuses on practical competitiveness. The central objective is to improve portfolio behavior in bull markets without abandoning risk control during stressed periods.

## 2. Problem Statement

Many portfolio strategies fail for one of two reasons. Some are too static and do not adapt to the market. Others adapt, but become so defensive that they sacrifice too much upside in favorable conditions. Version 1 of this project highlighted this exact trade-off. It delivered sensible risk control, but it did not always keep pace with simpler benchmark portfolios during prolonged growth periods. Therefore, a stronger framework is needed—one that adapts to market stress while still allowing growth assets to lead when conditions are supportive.

## 3. Aim

To build a more competitive adaptive portfolio management system that integrates regime detection, momentum tilting, trend filtering, volatility targeting, and defensive overlays into a single multi-asset allocation engine.

## 4. Objectives

1. To construct a diversified multi-asset ETF universe.
2. To engineer regime features from volatility, drawdown, correlation, and credit-style signals.
3. To classify markets into economically meaningful states.
4. To use tactical momentum to tilt allocations toward assets with stronger relative strength.
5. To penalize assets that break trend support.
6. To scale the portfolio toward a target volatility level.
7. To compare performance against static benchmarks and Version 1 logic.
8. To produce a full chart pack and metrics summary suitable for thesis, interview, and GitHub use.

## 5. Literature Review Direction

This section can be expanded with literature on:
- Modern Portfolio Theory,
- tactical asset allocation,
- trend following,
- regime switching,
- volatility targeting,
- risk parity,
- drawdown-aware portfolio construction,
- and machine learning in portfolio management.

## 6. Methodology Summary

The project works in six layers:
1. data ingestion,
2. feature engineering,
3. regime scoring,
4. strategic base allocation,
5. tactical overlay adjustment,
6. backtesting and evaluation.

## 7. Data Layer

The system downloads monthly data for a diversified ETF universe and additional signal assets. Prices are converted to month-end series, then transformed into returns and feature inputs.

## 8. Regime Layer

The regime model blends deterministic stress logic with an optional clustering overlay. The purpose is not to predict the future perfectly, but to classify the current market environment in a disciplined way. The final regime labels are:

- Calm Growth
- Volatile Growth
- Defensive / Uncertain
- Crisis / Stress

## 9. Tactical Layer

The model then adjusts strategic weights using:
- 3- and 6-month momentum,
- 10-month trend support,
- defensive penalties for weak assets,
- offensive scaling for strong assets.

## 10. Risk Layer

Risk is estimated through a covariance matrix. Portfolio volatility is then scaled toward a target annualized volatility, subject to max-weight and cash constraints. This helps maintain more stable portfolio risk across market states.

## 11. Results and Expected Interpretation

Compared with Version 1, Version 2 should ideally:
- participate better in bullish periods,
- reduce unnecessary defensiveness,
- preserve capital during stressed regimes,
- and deliver more compelling cumulative-return charts.

## 12. Discussion Themes

In the discussion chapter, the following themes are especially useful:
- how tactical overlays changed the return path,
- whether volatility targeting improved consistency,
- whether regime changes corresponded to macro-stress periods,
- whether momentum tilt caused concentration risk,
- how turnover smoothing balanced responsiveness and realism.

## 13. Limitation Themes

- reliance on historical ETF behavior,
- model sensitivity to momentum lookbacks,
- monthly rebalancing may miss intra-month crashes,
- no taxes or bid-ask spread modeling,
- signal set can be expanded with macro and news-aware features later.

## 14. Future Work

The next evolutionary step after V2 would be an event-aware portfolio engine incorporating:
- election windows,
- geopolitical stress indicators,
- inflation surprise regimes,
- credit-spread shocks,
- central-bank event dummies,
- or news sentiment.

That extension would move the project from market-reactive to partially event-aware.
