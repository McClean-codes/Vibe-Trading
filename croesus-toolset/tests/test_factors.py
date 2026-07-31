"""Tests for croesus_toolset.factors — alpha-zoo factor subset.

All tests use synthetic data; no network calls.
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv_df(n=300, seed=42):
    """Create a realistic OHLCV DataFrame."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n)
    returns = rng.normal(0.0005, 0.02, size=n)
    closes = 100.0 * np.cumprod(1 + returns)
    highs = closes * (1 + rng.uniform(0, 0.02, size=n))
    lows = closes * (1 - rng.uniform(0, 0.02, size=n))
    opens = closes * (1 + rng.normal(0, 0.005, size=n))
    volumes = rng.integers(10000, 100000, size=n).astype(float)
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": volumes,
    }, index=dates)


# ---------------------------------------------------------------------------
# Momentum factors
# ---------------------------------------------------------------------------

class TestMomentum:
    def test_momentum_12_1_returns_series(self):
        from croesus_toolset.factors.momentum import momentum_12_1
        df = _make_ohlcv_df(300)
        result = momentum_12_1(df)
        assert isinstance(result, pd.Series)
        assert result.name == "momentum_12_1"

    def test_momentum_12_1_nan_when_insufficient(self):
        from croesus_toolset.factors.momentum import momentum_12_1
        df = _make_ohlcv_df(10)  # too short
        result = momentum_12_1(df)
        assert result.isna().all()

    def test_momentum_12_1_expected_value(self):
        from croesus_toolset.factors.momentum import momentum_12_1
        # Constant price → momentum should be 0
        df = pd.DataFrame({
            "close": [100.0] * 300,
            "high": [101.0] * 300,
            "low": [99.0] * 300,
            "volume": [1000.0] * 300,
        })
        result = momentum_12_1(df)
        # After warmup, values should be 0 (no change)
        valid = result.dropna()
        assert (valid == 0.0).all()

    def test_roc_returns_series(self):
        from croesus_toolset.factors.momentum import roc
        df = _make_ohlcv_df(100)
        result = roc(df)
        assert isinstance(result, pd.Series)
        assert result.name == "roc"

    def test_roc_expected_value(self):
        from croesus_toolset.factors.momentum import roc
        # Constant price → ROC should be 0
        df = pd.DataFrame({
            "close": [100.0] * 50,
            "high": [101.0] * 50,
            "low": [99.0] * 50,
            "volume": [1000.0] * 50,
        })
        result = roc(df, period=10)
        valid = result.dropna()
        assert (valid == 0.0).all()


# ---------------------------------------------------------------------------
# Risk factors
# ---------------------------------------------------------------------------

class TestRisk:
    def test_volatility_30_returns_series(self):
        from croesus_toolset.factors.risk import volatility_30
        df = _make_ohlcv_df(100)
        result = volatility_30(df)
        assert isinstance(result, pd.Series)
        assert result.name == "volatility_30"

    def test_volatility_30_nan_when_insufficient(self):
        from croesus_toolset.factors.risk import volatility_30
        df = _make_ohlcv_df(5)
        result = volatility_30(df)
        assert result.isna().all()

    def test_volatility_30_positive(self):
        from croesus_toolset.factors.risk import volatility_30
        df = _make_ohlcv_df(100)
        result = volatility_30(df)
        valid = result.dropna()
        assert (valid >= 0).all()

    def test_volatility_30_constant_price_zero(self):
        from croesus_toolset.factors.risk import volatility_30
        df = pd.DataFrame({
            "close": [100.0] * 100,
            "volume": [1000.0] * 100,
        })
        result = volatility_30(df)
        valid = result.dropna()
        # Constant price → zero volatility
        assert (valid == 0.0).all()


# ---------------------------------------------------------------------------
# Reversal factors
# ---------------------------------------------------------------------------

class TestReversal:
    def test_mean_reversion_20_returns_series(self):
        from croesus_toolset.factors.reversal import mean_reversion_20
        df = _make_ohlcv_df(100)
        result = mean_reversion_20(df)
        assert isinstance(result, pd.Series)
        assert result.name == "mean_reversion_20"

    def test_mean_reversion_20_nan_when_insufficient(self):
        from croesus_toolset.factors.reversal import mean_reversion_20
        df = _make_ohlcv_df(5)
        result = mean_reversion_20(df)
        assert result.isna().all()

    def test_mean_reversion_20_zero_for_constant(self):
        from croesus_toolset.factors.reversal import mean_reversion_20
        df = pd.DataFrame({
            "close": [100.0] * 50,
            "volume": [1000.0] * 50,
        })
        result = mean_reversion_20(df)
        valid = result.dropna()
        assert (valid == 0.0).all()


# ---------------------------------------------------------------------------
# Alpha factors
# ---------------------------------------------------------------------------

class TestAlpha:
    def test_alpha_momentum_rank_returns_series(self):
        from croesus_toolset.factors.alpha import alpha_momentum_rank
        df = _make_ohlcv_df(300)
        result = alpha_momentum_rank(df)
        assert isinstance(result, pd.Series)
        assert result.name == "alpha_momentum_rank"

    def test_alpha_momentum_rank_in_0_1(self):
        from croesus_toolset.factors.alpha import alpha_momentum_rank
        df = _make_ohlcv_df(300)
        result = alpha_momentum_rank(df)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 1).all()

    def test_alpha_price_volume_divergence_returns_series(self):
        from croesus_toolset.factors.alpha import alpha_price_volume_divergence
        df = _make_ohlcv_df(100)
        result = alpha_price_volume_divergence(df)
        assert isinstance(result, pd.Series)
        assert result.name == "alpha_price_volume_divergence"

    def test_alpha_price_volume_divergence_range(self):
        from croesus_toolset.factors.alpha import alpha_price_volume_divergence
        df = _make_ohlcv_df(100)
        result = alpha_price_volume_divergence(df)
        valid = result.dropna()
        assert (valid >= -1).all()
        assert (valid <= 1).all()


# ---------------------------------------------------------------------------
# Technical factor wrappers
# ---------------------------------------------------------------------------

class TestTechnical:
    def test_rsi_14_series(self):
        from croesus_toolset.factors.technical import rsi_14
        df = _make_ohlcv_df(100)
        result = rsi_14(df)
        assert isinstance(result, pd.Series)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_macd_series(self):
        from croesus_toolset.factors.technical import macd
        df = _make_ohlcv_df(100)
        result = macd(df)
        assert isinstance(result, pd.Series)
        assert not result.dropna().empty

    def test_bollinger_width_series(self):
        from croesus_toolset.factors.technical import bollinger_width
        df = _make_ohlcv_df(100)
        result = bollinger_width(df)
        assert isinstance(result, pd.Series)
        valid = result.dropna()
        assert (valid >= 0).all()

    def test_obv_slope_series(self):
        from croesus_toolset.factors.technical import obv_slope
        df = _make_ohlcv_df(100)
        result = obv_slope(df)
        assert isinstance(result, pd.Series)
        # OBV slope can be positive or negative

    def test_vwap_deviation_series(self):
        from croesus_toolset.factors.technical import vwap_deviation
        df = _make_ohlcv_df(100)
        result = vwap_deviation(df)
        assert isinstance(result, pd.Series)


# ---------------------------------------------------------------------------
# Factor registry
# ---------------------------------------------------------------------------

class TestFactorRegistry:
    def test_all_factors_registered(self):
        from croesus_toolset.factors import FACTOR_REGISTRY
        expected = {
            "rsi_14", "macd", "bollinger_width",
            "momentum_12_1", "roc", "volatility_30",
            "mean_reversion_20", "alpha_momentum_rank",
            "alpha_price_volume_divergence", "obv_slope", "vwap_deviation",
        }
        assert set(FACTOR_REGISTRY.keys()) == expected

    def test_get_factor_unknown_raises(self):
        from croesus_toolset.factors import get_factor
        with pytest.raises(KeyError, match="Unknown factor"):
            get_factor("nonexistent")

    def test_get_factor_returns_callable(self):
        from croesus_toolset.factors import get_factor
        func = get_factor("rsi_14")
        assert callable(func)
