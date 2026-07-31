"""Technical indicator computation — pure Python (numpy/pandas).

Lifted from Vibe-Trading agent/src/tools/technical_indicator_tool.py.
All functions are pure math: take a pd.Series of close prices, return a value.
No network calls, no agent-runtime dependencies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Defaults ──────────────────────────────────────────────────────────────────
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2.0
SMA_PERIODS = (20, 50, 200)
EMA_PERIOD = 20


# ── SMA ──────────────────────────────────────────────────────────────────────

def compute_sma(close: pd.Series, period: int = 20) -> float | None:
    """Simple moving average over the last *period* bars."""
    if len(close) < period:
        return None
    return float(close.iloc[-period:].mean())


# ── EMA ──────────────────────────────────────────────────────────────────────

def compute_ema(close: pd.Series, period: int = 20) -> float | None:
    """Exponential moving average over the full series."""
    if len(close) < period:
        return None
    return float(close.ewm(span=period, adjust=False).mean().iloc[-1])


# ── RSI ──────────────────────────────────────────────────────────────────────

def compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> float | None:
    """Relative Strength Index (Wilder smoothing) over *period* bars.

    Returns a value in [0, 100], or None if insufficient data.
    """
    if len(close) < period + 1:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window=period).mean().iloc[-1]
    avg_loss = loss.rolling(window=period).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


# ── MACD ─────────────────────────────────────────────────────────────────────

def compute_macd(
    close: pd.Series,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
) -> dict[str, float] | None:
    """MACD line, signal line, and histogram.

    Returns a dict with keys macd_line, signal_line, histogram,
    or None if insufficient data.
    """
    if len(close) < slow + signal:
        return None
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return {
        "macd_line": round(float(macd_line.iloc[-1]), 4),
        "signal_line": round(float(signal_line.iloc[-1]), 4),
        "histogram": round(float(histogram.iloc[-1]), 4),
    }


# ── Bollinger Bands ──────────────────────────────────────────────────────────

def compute_bollinger(
    close: pd.Series,
    period: int = BB_PERIOD,
    num_std: float = BB_STD,
) -> dict[str, float] | None:
    """Bollinger Bands: upper, middle (SMA), lower.

    Returns a dict with keys upper, middle, lower,
    or None if insufficient data.
    """
    if len(close) < period:
        return None
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    return {
        "upper": round(float(sma.iloc[-1] + num_std * std.iloc[-1]), 2),
        "middle": round(float(sma.iloc[-1]), 2),
        "lower": round(float(sma.iloc[-1] - num_std * std.iloc[-1]), 2),
    }
