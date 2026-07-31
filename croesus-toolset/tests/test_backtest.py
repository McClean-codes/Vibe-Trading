"""Tests for croesus_toolset.backtest -- metrics and validation."""

import math
import numpy as np
import pandas as pd
import pytest


def _equity_curve(n=252, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=n)
    returns = rng.normal(0.0004, 0.01, size=n)
    equity = 100_000.0 * np.cumprod(1 + returns)
    return pd.Series(equity, index=dates)


class TestBarReturns:
    def test_bar_returns_simple(self):
        from croesus_toolset.backtest import bar_returns
        close = pd.Series([100.0, 101.0, 99.0, 102.0])
        result = bar_returns(close)
        assert len(result) == 3
        assert result[0] == pytest.approx(0.01, abs=1e-6)

    def test_bar_returns_first_is_nan(self):
        from croesus_toolset.backtest import bar_returns
        close = pd.Series([100.0, 100.0])
        result = bar_returns(close)
        assert len(result) == 1


class TestSharpeRatio:
    def test_sharpe_zero_std(self):
        from croesus_toolset.backtest import sharpe_ratio
        returns = pd.Series([0.01] * 20)
        assert sharpe_ratio(returns) == pytest.approx(0.0)

    def test_sharpe_positive_for_positive_returns(self):
        from croesus_toolset.backtest import sharpe_ratio
        rng = np.random.default_rng(99)
        returns = pd.Series(0.005 + rng.normal(0, 0.001, size=100))
        assert sharpe_ratio(returns) > 0


class TestMaxDrawdown:
    def test_max_drawdown_always_negative_or_zero(self):
        from croesus_toolset.backtest import max_drawdown
        assert max_drawdown(_equity_curve()) <= 0.0

    def test_max_drawdown_monotonic_returns_zero(self):
        from croesus_toolset.backtest import max_drawdown
        assert max_drawdown(pd.Series([100, 110, 120, 130])) == pytest.approx(0.0)

    def test_max_drawdown_known_case(self):
        from croesus_toolset.backtest import max_drawdown
        assert max_drawdown(pd.Series([100, 110, 90, 95])) == pytest.approx(-20/110, abs=1e-4)


class TestGapSafeBarReturns:
    def test_halted_asset_gap_detected(self):
        from croesus_toolset.backtest import gap_safe_bar_returns
        close = pd.Series([100.0]*7 + [105.0])
        result = gap_safe_bar_returns(close, halted_threshold=5)
        assert result.iloc[-1] == pytest.approx(0.05, abs=1e-6)

    def test_normal_series_unchanged(self):
        from croesus_toolset.backtest import gap_safe_bar_returns
        close = pd.Series([100.0, 101.0, 102.0, 103.0])
        result = gap_safe_bar_returns(close, halted_threshold=5)
        expected = close.pct_change().fillna(0.0)
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_halts_mask_returned(self):
        from croesus_toolset.backtest import gap_safe_bar_returns
        close = pd.Series([100.0]*6 + [105.0])
        result = gap_safe_bar_returns(close, halted_threshold=5)
        assert hasattr(result, 'attrs')


class TestSignSafeBenchmark:
    def test_near_zero_no_inf_nan(self):
        from croesus_toolset.backtest import sign_safe_benchmark
        close = pd.Series([100.0, 0.0001, 100.0, 200.0])
        result = sign_safe_benchmark(close)
        assert np.all(np.isfinite(result))

    def test_exact_zero_no_inf_nan(self):
        from croesus_toolset.backtest import sign_safe_benchmark
        close = pd.Series([100.0, 0.0, 100.0, 200.0])
        result = sign_safe_benchmark(close)
        assert np.all(np.isfinite(result))

    def test_normal_matches_pct_change(self):
        from croesus_toolset.backtest import sign_safe_benchmark
        close = pd.Series([100.0, 105.0, 110.0, 115.0])
        result = sign_safe_benchmark(close)
        expected = close.pct_change().fillna(0.0)
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_no_nan_in_output(self):
        from croesus_toolset.backtest import sign_safe_benchmark
        close = pd.Series([100.0, 0.0001, 0.0, 50.0, 200.0])
        assert not sign_safe_benchmark(close).isna().any()


class TestUsdmLiquidationPrice:
    def test_long_10x_near_54k(self):
        from croesus_toolset.backtest import usdm_liquidation_price
        liq = usdm_liquidation_price(60000.0, 10, "long", "isolated")
        assert liq == pytest.approx(54000.0, rel=0.02)

    def test_short_10x_above_entry(self):
        from croesus_toolset.backtest import usdm_liquidation_price
        assert usdm_liquidation_price(60000.0, 10, "short", "isolated") > 60000.0

    def test_cross_margin_deterministic(self):
        from croesus_toolset.backtest import usdm_liquidation_price
        assert usdm_liquidation_price(60000.0, 10, "long", "cross") < 60000.0

    def test_higher_leverage_closer(self):
        from croesus_toolset.backtest import usdm_liquidation_price
        assert usdm_liquidation_price(60000.0, 20, "long", "isolated") > usdm_liquidation_price(60000.0, 10, "long", "isolated")

    def test_invalid_side_raises(self):
        from croesus_toolset.backtest import usdm_liquidation_price
        with pytest.raises(ValueError):
            usdm_liquidation_price(60000.0, 10, "sideways", "isolated")

    def test_invalid_margin_type_raises(self):
        from croesus_toolset.backtest import usdm_liquidation_price
        with pytest.raises(ValueError):
            usdm_liquidation_price(60000.0, 10, "long", "portfolio")
