"""Tests for croesus_toolset.loaders — data-source adapters.

All tests use mocked data; no network calls.
"""

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv_df(n=100):
    """Create a small inline OHLCV DataFrame for testing."""
    import numpy as np
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2025-01-01", periods=n)
    closes = 100.0 + np.cumsum(rng.normal(0, 1, size=n))
    return pd.DataFrame({
        "open": closes - 1,
        "high": closes + 2,
        "low": closes - 2,
        "close": closes,
        "volume": rng.integers(1000, 10000, size=n),
    }, index=dates)


# ---------------------------------------------------------------------------
# BaseLoader
# ---------------------------------------------------------------------------

class TestBaseLoader:
    def test_cannot_instantiate_directly(self):
        from croesus_toolset.loaders.base import BaseLoader
        with pytest.raises(TypeError):
            BaseLoader()


# ---------------------------------------------------------------------------
# get_loader
# ---------------------------------------------------------------------------

class TestGetLoader:
    def test_get_yfinance(self):
        from croesus_toolset.loaders import get_loader
        loader = get_loader("yfinance")
        assert type(loader).__name__ == "YFinanceLoader"

    def test_get_ccxt(self):
        from croesus_toolset.loaders import get_loader
        loader = get_loader("ccxt")
        assert type(loader).__name__ == "CcxtLoader"

    def test_unknown_loader_raises(self):
        from croesus_toolset.loaders import get_loader
        with pytest.raises(ValueError, match="Unknown loader"):
            get_loader("nonexistent")


# ---------------------------------------------------------------------------
# YFinanceLoader
# ---------------------------------------------------------------------------

class TestYFinanceLoader:
    def test_fetch_ohlcv_happy_path(self):
        """Test that YFinanceLoader normalises column names."""
        from unittest.mock import MagicMock, patch
        import sys
        from croesus_toolset.loaders.yfinance_loader import YFinanceLoader

        df = _make_ohlcv_df()
        # yfinance returns uppercase columns
        df_upper = df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume",
        })

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = df_upper
        mock_yf = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker

        # Patch yfinance in sys.modules so the import inside fetch_ohlcv works
        old = sys.modules.get("yfinance")
        sys.modules["yfinance"] = mock_yf
        try:
            loader = YFinanceLoader()
            result = loader.fetch_ohlcv("BTC-USD", interval="1d", window="90d")
        finally:
            if old is not None:
                sys.modules["yfinance"] = old
            else:
                sys.modules.pop("yfinance", None)

        assert len(result) == 100
        assert "close" in result.columns
        assert result.index[0] == pd.Timestamp("2025-01-01")

    def test_fetch_ohlcv_empty_raises(self):
        from unittest.mock import MagicMock
        import sys
        from croesus_toolset.loaders.yfinance_loader import YFinanceLoader

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_yf = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker

        old = sys.modules.get("yfinance")
        sys.modules["yfinance"] = mock_yf
        try:
            loader = YFinanceLoader()
            with pytest.raises(ValueError, match="No data"):
                loader.fetch_ohlcv("INVALID", interval="1d")
        finally:
            if old is not None:
                sys.modules["yfinance"] = old
            else:
                sys.modules.pop("yfinance", None)

    def test_fetch_ohlcv_missing_import(self):
        import sys
        from croesus_toolset.loaders.yfinance_loader import YFinanceLoader

        saved = sys.modules.pop("yfinance", None)
        # Make import fail by setting to None
        sys.modules["yfinance"] = None
        try:
            loader = YFinanceLoader()
            with pytest.raises(ImportError, match="yfinance is required"):
                loader.fetch_ohlcv("BTC-USD")
        finally:
            if saved is not None:
                sys.modules["yfinance"] = saved
            else:
                sys.modules.pop("yfinance", None)


# ---------------------------------------------------------------------------
# CcxtLoader
# ---------------------------------------------------------------------------

class TestCcxtLoader:
    def test_fetch_ohlcv_happy_path(self):
        from unittest.mock import MagicMock
        import sys
        from croesus_toolset.loaders.ccxt_loader import CcxtLoader

        # ccxt returns [timestamp_ms, o, h, l, c, v]
        ts = 1704067200000  # 2024-01-01 00:00 UTC
        raw = [
            [ts + i * 86400000, 100 + i, 105 + i, 95 + i, 102 + i, 1000 + i]
            for i in range(5)
        ]

        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = raw
        mock_ccxt = MagicMock()
        mock_ccxt.binance.return_value = mock_exchange

        old = sys.modules.get("ccxt")
        sys.modules["ccxt"] = mock_ccxt
        try:
            loader = CcxtLoader()
            result = loader.fetch_ohlcv(
                "BTC/USDT", interval="1d", window="5d", exchange="binance"
            )
        finally:
            if old is not None:
                sys.modules["ccxt"] = old
            else:
                sys.modules.pop("ccxt", None)

        assert len(result) == 5
        assert "close" in result.columns
        assert result["close"].iloc[0] == 102

    def test_fetch_ohlcv_empty_raises(self):
        from unittest.mock import MagicMock
        import sys
        from croesus_toolset.loaders.ccxt_loader import CcxtLoader

        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = []
        mock_ccxt = MagicMock()
        mock_ccxt.binance.return_value = mock_exchange

        old = sys.modules.get("ccxt")
        sys.modules["ccxt"] = mock_ccxt
        try:
            loader = CcxtLoader()
            with pytest.raises(ValueError, match="No data"):
                loader.fetch_ohlcv("BTC/USDT", exchange="binance")
        finally:
            if old is not None:
                sys.modules["ccxt"] = old
            else:
                sys.modules.pop("ccxt", None)

    def test_fetch_ohlcv_unknown_exchange(self):
        import sys
        from unittest.mock import MagicMock
        from croesus_toolset.loaders.ccxt_loader import CcxtLoader

        class FakeCcxt:
            binance = MagicMock()
        mock_ccxt = FakeCcxt()

        old = sys.modules.get("ccxt")
        sys.modules["ccxt"] = mock_ccxt
        try:
            loader = CcxtLoader()
            with pytest.raises(ValueError, match="Unknown exchange"):
                loader.fetch_ohlcv("BTC/USDT", exchange="nonexistent")
        finally:
            if old is not None:
                sys.modules["ccxt"] = old
            else:
                sys.modules.pop("ccxt", None)

    def test_fetch_ohlcv_missing_import(self):
        import sys
        from croesus_toolset.loaders.ccxt_loader import CcxtLoader

        saved = sys.modules.pop("ccxt", None)
        sys.modules["ccxt"] = None
        try:
            loader = CcxtLoader()
            with pytest.raises(ImportError, match="ccxt is required"):
                loader.fetch_ohlcv("BTC/USDT", exchange="binance")
        finally:
            if saved is not None:
                sys.modules["ccxt"] = saved
            else:
                sys.modules.pop("ccxt", None)
