"""Risk factors — lifted from agent/src/factors/zoo/.

Attribution: Adapted from Microsoft Qlib alpha factors (Apache-2.0).
Copyright (c) Microsoft Corporation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def volatility_30(df: pd.DataFrame, **params) -> pd.Series:
    """Realised volatility over *period* bars.

    Computed as the rolling standard deviation of log returns, annualised
    by sqrt(252) for daily data.
    """
    close = df["close"]
    period = params.get("period", 30)
    annualise = params.get("annualise", True)

    if len(close) < period + 1:
        return pd.Series(np.nan, index=close.index, name="volatility_30")

    log_ret = np.log(close / close.shift(1))
    vol = log_ret.rolling(window=period, min_periods=period).std()
    if annualise:
        vol = vol * np.sqrt(252)

    return vol.rename("volatility_30")
