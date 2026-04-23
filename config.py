from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProjectConfig:
    # Core dates
    start_date: str = "2010-01-01"
    end_date: str = "2026-01-01"

    # Portfolio assets
    tickers: list[str] = field(default_factory=lambda: [
        "SPY",  # US equities
        "EFA",  # Developed ex-US
        "EEM",  # Emerging markets
        "TLT",  # Long treasuries
        "IEF",  # Intermediate treasuries
        "GLD",  # Gold
        "DBC",  # Broad commodities
        "VNQ",  # Real estate
        "BIL",  # Cash proxy
    ])

    # Signal-only tickers
    signal_tickers: list[str] = field(default_factory=lambda: [
        "^VIX",  # implied volatility
        "HYG",   # high yield credit
        "LQD",   # investment grade credit
    ])

    # Data frequency / windows
    resample_freq: str = "ME"
    lookback_months: int = 12
    short_momentum_months: int = 3
    long_momentum_months: int = 6
    trend_window_months: int = 10

    # Risk / optimization
    covariance_method: str = "shrinkage"   # sample / ewma / shrinkage
    ewma_lambda: float = 0.94
    annual_risk_free_rate: float = 0.02
    annual_target_volatility: float = 0.10

    # Portfolio construction controls
    min_weight: float = 0.0
    max_weight: float = 0.40
    cash_ticker: str = "BIL"
    base_cash_floor: float = 0.03
    stress_cash_floor: float = 0.20
    extreme_stress_cash_floor: float = 0.35
    turnover_penalty: float = 0.20
    transaction_cost_bps: float = 10.0
    smoothing_alpha: float = 0.55

    # Regime logic
    n_regimes: int = 4
    random_state: int = 42
    use_cluster_overlay: bool = True

    # Tactical overlays
    momentum_strength: float = 0.35
    reversal_penalty_strength: float = 0.30
    trend_floor_multiplier: float = 0.25
    defensive_regime_multiplier: float = 0.65
    offensive_regime_multiplier: float = 1.20

    # Benchmarks
    benchmark_6040_equity_ticker: str = "SPY"
    benchmark_6040_bond_ticker: str = "IEF"

    # Outputs
    output_dir: str = "outputs"


CONFIG = ProjectConfig()

ASSET_NAME_MAP = {
    "SPY": "US Equities",
    "EFA": "Developed ex-US",
    "EEM": "Emerging Markets",
    "TLT": "Long-Term Bonds",
    "IEF": "Intermediate Bonds",
    "GLD": "Gold",
    "DBC": "Commodities",
    "VNQ": "Real Estate",
    "BIL": "Cash / T-Bills",
}

REGIME_NAME_MAP = {
    0: "Calm Growth",
    1: "Volatile Growth",
    2: "Defensive / Uncertain",
    3: "Crisis / Stress",
}
