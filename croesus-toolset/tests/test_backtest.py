"""Tests for croesus_toolset.backtest — metrics and validation."""

import math

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _equity_curve(n: int = 252, seed: int = 42) -> pd.Series:
    """Synthetic equity curve (deterministic)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=n)
    returns = rng.normal(0.0004, 0.01, size=n)
    equity = 100_000.0 * np.cumprod(1 + returns)
    return pd.Series(equity, index=dates)


# ---------------------------------------------------------------------------
# bar_returns
# ---------------------------------------------------------------------------

class TestBarReturns:
    def test_bar_returns_simple(self):
        from croesus_toolset.backtest import bar_returns
        close = pd.Series([100.0, 101.0, 99.0, 102.0])
        result = bar_returns(close)
        # pct_change: (101-100)/100=0.01, (99-101)/101≈-0.0198, (102-99)/99≈0.0303
        expected = [0.01, (99 - 101) / 101, (102 - 99) / 99]
        assert len(result) == 3
        for r, e in zip(result, expected):
            assert r == pytest.approx(e, abs=1e-6)

    def test_bar_returns_first_is_nan(self):
        from croesus_toolset.backtest import bar_returns
        close = pd.Series([100.0, 100.0])
        result = bar_returns(close)
        assert len(result) == 1
        assert math.isfinite(result[0]) or math.isnan(result[0])


# ---------------------------------------------------------------------------
# sharpe_ratio
# ---------------------------------------------------------------------------

class TestSharpeRatio:
    def test_sharpe_zero_std(self):
        """Constant returns → Sharpe = 0 (division by near-zero std)."""
        from croesus_toolset.backtest import sharpe_ratio
        returns = pd.Series([0.01] * 20)
        result = sharpe_ratio(returns)
        assert result == pytest.approx(0.0)

    def test_sharpe_positive_for_positive_returns(self):
        from croesus_toolset.backtest import sharpe_ratio
        rng = np.random.default_rng(99)
        returns = pd.Series(0.005 + rng.normal(0, 0.001, size=100))
        result = sharpe_ratio(returns)
        assert result > 0

    def test_sharpe_negative_for_negative_returns(self):
        from croesus_toolset.backtest import sharpe_ratio
        rng = np.random.default_rng(99)
        returns = pd.Series(-0.005 + rng.normal(0, 0.001, size=100))
        result = sharpe_ratio(returns)
        assert result < 0


# ---------------------------------------------------------------------------
# max_drawdown
# ---------------------------------------------------------------------------

class TestMaxDrawdown:
    def test_max_drawdown_always_negative_or_zero(self):
        from croesus_toolset.backtest import max_drawdown
        eq = _equity_curve()
        result = max_drawdown(eq)
        assert result <= 0.0

    def test_max_drawdown_on_monotonic_returns_zero(self):
        from croesus_toolset.backtest import max_drawdown
        eq = pd.Series([100, 110, 120, 130])
        result = max_drawdown(eq)
        assert result == pytest.approx(0.0)

    def test_max_drawdown_known_case(self):
        from croesus_toolset.backtest import max_drawdown
        eq = pd.Series([100, 110, 90, 95])
        # Peak 110, trough 90 → dd = (90-110)/110 = -0.1818...
        result = max_drawdown(eq)
        assert result == pytest.approx(-20 / 110, abs=1e-4)
