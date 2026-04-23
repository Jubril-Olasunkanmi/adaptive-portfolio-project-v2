from __future__ import annotations

import pandas as pd
import yfinance as yf


def _extract_close_prices(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        raise ValueError("No data downloaded from yfinance.")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Adj Close" in raw.columns.get_level_values(0):
            prices = raw["Adj Close"].copy()
        elif "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"].copy()
        else:
            prices = raw.xs(raw.columns.get_level_values(0)[0], axis=1, level=0).copy()
    else:
        prices = raw.copy()
    return prices.sort_index()


def download_price_panel(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    raw = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
        group_by="column",
    )
    prices = _extract_close_prices(raw)
    prices = prices.dropna(how="all").ffill()
    return prices


def prepare_monthly_data(
    prices: pd.DataFrame,
    freq: str = "ME",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly_prices = prices.resample(freq).last().dropna(how="all")
    monthly_returns = monthly_prices.pct_change().dropna(how="all")
    return monthly_prices, monthly_returns


def align_portfolio_and_signal_data(
    portfolio_prices: pd.DataFrame,
    signal_prices: pd.DataFrame,
    freq: str = "ME",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    monthly_portfolio_prices, monthly_portfolio_returns = prepare_monthly_data(portfolio_prices, freq=freq)
    monthly_signal_prices, monthly_signal_returns = prepare_monthly_data(signal_prices, freq=freq)

    common_index = monthly_portfolio_returns.index.intersection(monthly_signal_returns.index)
    monthly_portfolio_prices = monthly_portfolio_prices.loc[common_index]
    monthly_portfolio_returns = monthly_portfolio_returns.loc[common_index]
    monthly_signal_prices = monthly_signal_prices.loc[common_index]
    monthly_signal_returns = monthly_signal_returns.loc[common_index]

    return (
        monthly_portfolio_prices,
        monthly_portfolio_returns,
        monthly_signal_prices,
        monthly_signal_returns,
    )
