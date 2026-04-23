from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def sample_covariance(returns_window: pd.DataFrame) -> pd.DataFrame:
    return returns_window.cov()


def ewma_covariance(returns_window: pd.DataFrame, lam: float = 0.94) -> pd.DataFrame:
    x = returns_window.values
    n_obs, n_assets = x.shape
    mean_adj = x - np.nanmean(x, axis=0, keepdims=True)
    weights = np.array([(1.0 - lam) * (lam ** (n_obs - 1 - i)) for i in range(n_obs)])
    weights /= weights.sum()

    cov = np.zeros((n_assets, n_assets))
    for i in range(n_obs):
        row = mean_adj[i].reshape(-1, 1)
        cov += weights[i] * (row @ row.T)

    return pd.DataFrame(cov, index=returns_window.columns, columns=returns_window.columns)


def shrinkage_covariance(returns_window: pd.DataFrame) -> pd.DataFrame:
    lw = LedoitWolf()
    lw.fit(returns_window.values)
    cov = lw.covariance_
    return pd.DataFrame(cov, index=returns_window.columns, columns=returns_window.columns)


def estimate_covariance(returns_window: pd.DataFrame, method: str, ewma_lambda: float = 0.94) -> pd.DataFrame:
    method = method.lower()
    if method == "sample":
        return sample_covariance(returns_window)
    if method == "ewma":
        return ewma_covariance(returns_window, lam=ewma_lambda)
    if method == "shrinkage":
        return shrinkage_covariance(returns_window)
    raise ValueError(f"Unknown covariance method: {method}")
