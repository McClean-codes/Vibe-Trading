"""Backtest metrics -- bar returns, Sharpe ratio, max drawdown, gap-safe
returns, sign-safe benchmarks, USD-M liquidation price.

Lifted from Vibe-Trading agent/backtest/metrics.py,
agent/backtest/benchmark.py, and agent/backtest/perpetual_risk.py.
Pure math: takes pd.Series, returns float/Series. No I/O, no agent-runtime deps.
"""

from __future__ import annotations

import logging
import math
from typing import Literal

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)


def bar_returns(close: pd.Series) -> list[float]:
    """Compute simple bar-to-bar returns from a close-price series."""
    returns = close.pct_change().dropna()
    return returns.tolist()


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    """Annualised Sharpe ratio from a return series."""
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free
    std = float(excess.std())
    if std < 1e-12 or math.isnan(std):
        return 0.0
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    """Maximum drawdown from an equity curve."""
    if len(equity) < 2:
        return 0.0
    peak = equity.cummax()
    drawdown = (equity - peak) / peak.replace(0, np.nan)
    result = float(drawdown.min())
    return result if math.isfinite(result) else 0.0


def gap_safe_bar_returns(close: pd.Series, halted_threshold: int = 5) -> pd.Series:
    """Per-bar returns with gap-safe handling for halted assets."""
    prev = close.ffill().shift(1)
    usable_prev = np.isfinite(prev) & (prev > 0)
    positive_prev = prev.where(usable_prev)
    ret = close / positive_prev - 1
    ret = ret.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    price_diff = close.diff().fillna(0)
    same_as_prev = price_diff == 0
    streak = same_as_prev.groupby((same_as_prev != same_as_prev.shift()).cumsum()).cumsum()
    halted_mask = streak >= halted_threshold
    ret.attrs["halted_mask"] = halted_mask
    return ret


def sign_safe_benchmark(close: pd.Series) -> pd.Series:
    """Buy-and-hold compounded return series with sign-safe guards."""
    prev = close.ffill().shift(1)
    usable_prev = np.isfinite(prev) & (prev > 0)
    positive_prev = prev.where(usable_prev)
    undefined = int((prev.notna() & ~usable_prev).to_numpy().sum())
    if undefined:
        _log.warning("sign_safe_benchmark: %d bar(s) follow non-positive prior; return=0.0", undefined)
    ret = close / positive_prev - 1
    return ret.replace([np.inf, -np.inf], np.nan).fillna(0.0)


_DEFAULT_MAINTENANCE_RATE = 0.004


def usdm_liquidation_price(
    entry_price: float,
    leverage: int | float,
    side: Literal["long", "short"],
    margin_type: Literal["isolated", "cross"],
    maintenance_rate: float = _DEFAULT_MAINTENANCE_RATE,
) -> float:
    """Deterministic liquidation price for a USD-M perpetual position."""
    if entry_price <= 0 or not math.isfinite(entry_price):
        raise ValueError("entry_price must be positive and finite")
    if leverage < 1 or not math.isfinite(leverage):
        raise ValueError("leverage must be >= 1 and finite")
    if side not in ("long", "short"):
        raise ValueError("side must be 'long' or 'short'")
    if margin_type not in ("isolated", "cross"):
        raise ValueError("margin_type must be 'isolated' or 'cross'")
    if not (0 <= maintenance_rate < 1):
        raise ValueError("maintenance_rate must be in [0, 1)")
    if side == "long":
        return entry_price * (1.0 - 1.0 / leverage + maintenance_rate)
    else:
        return entry_price * (1.0 + 1.0 / leverage - maintenance_rate)
