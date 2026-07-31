"""Alpha factors — custom alpha signals.

These combine multiple OHLCV columns to produce alpha signals that go
beyond single-series technical indicators.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def alpha_momentum_rank(df: pd.DataFrame, **params) -> pd.Series:
    """Cross-sectional rank of recent momentum.

    Computes ``roc_20`` then ranks it as a percentile [0, 1].
    In single-asset mode this just normalises the ROC to [0, 1] using a
    CDF-like transform (tanh of z-scored ROC).
    """
    close = df["close"]
    period = params.get("period", 20)

    if len(close) < period + 1:
        return pd.Series(np.nan, index=close.index, name="alpha_momentum_rank")

    roc = close / close.shift(period) - 1.0
    mu = roc.rolling(window=252, min_periods=period).mean()
    sigma = roc.rolling(window=252, min_periods=period).std()
    z = (roc - mu) / sigma.replace(0, np.nan)
    # Sigmoidal rank: maps z-score to [0, 1]
    rank = 1.0 / (1.0 + np.exp(-z))
    return rank.rename("alpha_momentum_rank")


def alpha_price_volume_divergence(df: pd.DataFrame, **params) -> pd.Series:
    """Price-volume divergence signal.

    ``pv_div = rolling_corr(close, volume, 20)``

    A negative correlation (price rising, volume falling) suggests weakening
    conviction — potential reversal.
    """
    close = df["close"]
    volume = df["volume"]
    period = params.get("period", 20)

    if len(close) < period:
        return pd.Series(
            np.nan, index=close.index, name="alpha_price_volume_divergence"
        )

    corr = close.rolling(window=period, min_periods=period).corr(volume)
    return corr.rename("alpha_price_volume_divergence")
