"""Technical factor wrappers — thin wrappers around the indicators module
that return pd.Series for factor-registry compatibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi_14(df: pd.DataFrame, **params) -> pd.Series:
    """RSI as a pd.Series (not a scalar)."""
    from croesus_toolset.indicators import compute_rsi

    close = df["close"]
    period = params.get("period", 14)

    if len(close) < period + 1:
        return pd.Series(np.nan, index=close.index, name="rsi_14")

    # Compute rolling RSI
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.rename("rsi_14")


def macd(df: pd.DataFrame, **params) -> pd.Series:
    """MACD histogram as a pd.Series."""
    close = df["close"]
    fast = params.get("fast", 12)
    slow = params.get("slow", 26)
    signal = params.get("signal", 9)

    if len(close) < slow + signal:
        return pd.Series(np.nan, index=close.index, name="macd")

    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return histogram.rename("macd")


def bollinger_width(df: pd.DataFrame, **params) -> pd.Series:
    """Bollinger Band width: ``(upper - lower) / middle``."""
    close = df["close"]
    period = params.get("period", 20)
    num_std = params.get("num_std", 2.0)

    if len(close) < period:
        return pd.Series(np.nan, index=close.index, name="bollinger_width")

    sma = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    width = (upper - lower) / sma.replace(0, np.nan)
    return width.rename("bollinger_width")


def obv_slope(df: pd.DataFrame, **params) -> pd.Series:
    """On-Balance Volume slope over *period* bars.

    ``obv_slope = OBV[t] - OBV[t-period]`` normalised by period.
    """
    close = df["close"]
    volume = df["volume"]
    period = params.get("period", 20)

    if len(close) < period + 1:
        return pd.Series(np.nan, index=close.index, name="obv_slope")

    direction = np.sign(close.diff()).fillna(0)
    obv = (direction * volume).cumsum()
    slope = (obv - obv.shift(period)) / period
    return slope.rename("obv_slope")


def vwap_deviation(df: pd.DataFrame, **params) -> pd.Series:
    """Deviation from VWAP: ``(close - vwap) / vwap``.

    Uses typical price as VWAP proxy when volume-weighted price is unavailable.
    """
    close = df["close"]
    volume = df["volume"]
    period = params.get("period", 20)

    if len(close) < period:
        return pd.Series(np.nan, index=close.index, name="vwap_deviation")

    # Typical price as proxy
    high = df.get("high", close)
    low = df.get("low", close)
    typical = (high + low + close) / 3.0

    vwap = (typical * volume).rolling(window=period, min_periods=period).sum() / (
        volume.rolling(window=period, min_periods=period).sum()
    )
    dev = (close - vwap) / vwap.replace(0, np.nan)
    return dev.rename("vwap_deviation")
