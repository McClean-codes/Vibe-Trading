"""Reversal factors — lifted from agent/src/factors/zoo/.

Attribution: Adapted from Microsoft Qlib alpha factors (Apache-2.0).
Copyright (c) Microsoft Corporation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mean_reversion_20(df: pd.DataFrame, **params) -> pd.Series:
    """20-day mean reversion signal.

    Computes ``(close - sma_20) / sma_20`` — normalised deviation from the
    20-day moving average.  Negative values suggest the price is below the
    moving average (potential mean reversion buy).
    """
    close = df["close"]
    period = params.get("period", 20)

    if len(close) < period:
        return pd.Series(np.nan, index=close.index, name="mean_reversion_20")

    sma = close.rolling(window=period, min_periods=period).mean()
    return ((close - sma) / sma).rename("mean_reversion_20")
