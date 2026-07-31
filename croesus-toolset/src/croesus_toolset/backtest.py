"""Backtest metrics — bar returns, Sharpe ratio, max drawdown.

Lifted from Vibe-Trading agent/backtest/metrics.py (subset).
Pure math: takes pd.Series, returns float. No I/O, no agent-runtime deps.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def bar_returns(close: pd.Series) -> list[float]:
    """Compute simple bar-to-bar returns from a close-price series.

    Returns a list of length len(close) - 1. No leading NaN — the first
    return is (close[1] - close[0]) / close[0].
    """
    returns = close.pct_change().dropna()
    return returns.tolist()


def sharpe_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualised Sharpe ratio from a return series.

    When the standard deviation is zero (constant returns), the Sharpe
    is defined as 0.0 (no excess return per unit of risk).
    """
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free
    std = float(excess.std())
    if std < 1e-12 or math.isnan(std):
        return 0.0
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    """Maximum drawdown from an equity curve.

    Returns a negative float (or 0.0 if monotonic). Defined as the
    largest peak-to-trough decline as a fraction of the peak.
    """
    if len(equity) < 2:
        return 0.0
    peak = equity.cummax()
    drawdown = (equity - peak) / peak.replace(0, np.nan)
    result = float(drawdown.min())
    return result if math.isfinite(result) else 0.0
