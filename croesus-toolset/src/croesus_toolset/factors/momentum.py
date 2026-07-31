"""Momentum factors — lifted from agent/src/factors/zoo/.

Attribution: Adapted from Microsoft Qlib alpha factors (Apache-2.0).
Copyright (c) Microsoft Corporation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def momentum_12_1(df: pd.DataFrame, **params) -> pd.Series:
    """12-month momentum with 1-month reversal skip.

    ``momentum_12_1 = close[t-21] / close[t-252] - 1``

    Common in cross-sectional equity models; the 1-month skip avoids the
    well-documented short-term reversal effect.
    """
    close = df["close"]
    skip = params.get("skip", 21)  # ~1 month
    lookback = params.get("lookback", 252)  # ~12 months

    if len(close) < lookback + skip:
        return pd.Series(np.nan, index=close.index, name="momentum_12_1")

    return (close.shift(skip) / close.shift(lookback) - 1.0).rename(
        "momentum_12_1"
    )


def roc(df: pd.DataFrame, **params) -> pd.Series:
    """Rate of change over *period* bars.

    ``roc = close / close.shift(period) - 1``
    """
    close = df["close"]
    period = params.get("period", 20)

    if len(close) < period + 1:
        return pd.Series(np.nan, index=close.index, name="roc")

    return (close / close.shift(period) - 1.0).rename("roc")
