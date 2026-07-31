"""Tests for croesus_toolset.indicators — pure-math indicator functions."""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers to build synthetic close-price series
# ---------------------------------------------------------------------------

def _trending_close(n: int = 100, start: float = 100.0, drift: float = 0.5) -> pd.Series:
    """Linearly trending close series (deterministic)."""
    dates = pd.bdate_range("2025-01-01", periods=n)
    closes = start + np.arange(n) * drift
    return pd.Series(closes, index=dates, name="close")


def _volatile_close(n: int = 200, seed: int = 42) -> pd.Series:
    """Random-walk close series (deterministic seed)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=n)
    returns = rng.normal(0.0005, 0.02, size=n)
    prices = 100.0 * np.cumprod(1 + returns)
    return pd.Series(prices, index=dates, name="close")


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

class TestRSI:
    def test_rsi_returns_value_between_0_and_100(self):
        from croesus_toolset.indicators import compute_rsi
        close = _volatile_close(200)
        result = compute_rsi(close, period=14)
        assert result is not None
        assert 0.0 <= result <= 100.0

    def test_rsi_none_on_insufficient_data(self):
        from croesus_toolset.indicators import compute_rsi
        close = _trending_close(5)
        result = compute_rsi(close, period=14)
        assert result is None

    def test_rsi_100_on_strong_uptrend(self):
        """All gains, no losses → RSI should be 100."""
        from croesus_toolset.indicators import compute_rsi
        # Perfect uptrend: every bar is higher
        close = pd.Series(np.linspace(1, 2, 50))
        result = compute_rsi(close, period=14)
        assert result == pytest.approx(100.0)

    def test_rsi_0_on_strong_downtrend(self):
        """All losses, no gains → RSI should be 0."""
        from croesus_toolset.indicators import compute_rsi
        close = pd.Series(np.linspace(2, 1, 50))
        result = compute_rsi(close, period=14)
        assert result == pytest.approx(0.0)

    def test_rsi_default_period_14(self):
        from croesus_toolset.indicators import compute_rsi
        close = _volatile_close(200)
        rsi_default = compute_rsi(close)
        rsi_14 = compute_rsi(close, period=14)
        assert rsi_default == pytest.approx(rsi_14)


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

class TestMACD:
    def test_macd_returns_dict_with_three_keys(self):
        from croesus_toolset.indicators import compute_macd
        close = _volatile_close(200)
        result = compute_macd(close)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"macd_line", "signal_line", "histogram"}

    def test_macd_none_on_insufficient_data(self):
        from croesus_toolset.indicators import compute_macd
        close = _trending_close(10)
        result = compute_macd(close)
        assert result is None

    def test_macd_values_are_finite(self):
        from croesus_toolset.indicators import compute_macd
        close = _volatile_close(200)
        result = compute_macd(close)
        for v in result.values():
            assert np.isfinite(v)

    def test_macd_histogram_is_difference(self):
        from croesus_toolset.indicators import compute_macd
        close = _volatile_close(200)
        result = compute_macd(close)
        # Each value is independently rounded to 4dp, so allow small rounding diff
        assert result["histogram"] == pytest.approx(
            result["macd_line"] - result["signal_line"], abs=1e-3
        )


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

class TestBollinger:
    def test_bollinger_returns_dict_with_three_keys(self):
        from croesus_toolset.indicators import compute_bollinger
        close = _volatile_close(200)
        result = compute_bollinger(close)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"upper", "middle", "lower"}

    def test_bollinger_none_on_insufficient_data(self):
        from croesus_toolset.indicators import compute_bollinger
        close = _trending_close(5)
        result = compute_bollinger(close)
        assert result is None

    def test_bollinger_middle_is_sma(self):
        from croesus_toolset.indicators import compute_bollinger
        from croesus_toolset.indicators import compute_sma
        close = _volatile_close(200)
        bb = compute_bollinger(close, period=20)
        sma = compute_sma(close, period=20)
        assert bb["middle"] == pytest.approx(sma, abs=0.01)

    def test_bollinger_ordering_upper_gt_middle_gt_lower(self):
        from croesus_toolset.indicators import compute_bollinger
        close = _volatile_close(200)
        bb = compute_bollinger(close)
        assert bb["upper"] > bb["middle"] > bb["lower"]


# ---------------------------------------------------------------------------
# SMA / EMA
# ---------------------------------------------------------------------------

class TestSMA:
    def test_sma_returns_float(self):
        from croesus_toolset.indicators import compute_sma
        close = _trending_close(200)
        result = compute_sma(close, period=20)
        assert isinstance(result, float)

    def test_sma_none_on_insufficient_data(self):
        from croesus_toolset.indicators import compute_sma
        close = _trending_close(5)
        result = compute_sma(close, period=20)
        assert result is None

    def test_sma_is_mean_of_last_n(self):
        from croesus_toolset.indicators import compute_sma
        close = _trending_close(50)
        result = compute_sma(close, period=10)
        expected = float(close.iloc[-10:].mean())
        assert result == pytest.approx(expected)


class TestEMA:
    def test_ema_returns_float(self):
        from croesus_toolset.indicators import compute_ema
        close = _volatile_close(200)
        result = compute_ema(close, period=20)
        assert isinstance(result, float)

    def test_ema_none_on_insufficient_data(self):
        from croesus_toolset.indicators import compute_ema
        close = _trending_close(5)
        result = compute_ema(close, period=20)
        assert result is None
