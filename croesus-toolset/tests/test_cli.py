"""Tests for croesus_toolset.cli — CLI entrypoint.

Tests the new loader-based architecture with --loader, --period, and fetch-factors.
"""

import json
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_ohlcv_df(n=100, uppercase=False):
    """Create a mock OHLCV DataFrame."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2025-01-01", periods=n)
    closes = 100.0 + np.cumsum(rng.normal(0, 1, size=n))
    cols = {
        ("Open" if uppercase else "open"): closes - 1,
        ("High" if uppercase else "high"): closes + 2,
        ("Low" if uppercase else "low"): closes - 2,
        ("Close" if uppercase else "close"): closes,
        ("Volume" if uppercase else "volume"): rng.integers(1000, 10000, size=n),
    }
    return pd.DataFrame(cols, index=dates)


def _mock_loader_df():
    """Standard lowercase OHLCV for loader-based tests."""
    return _mock_ohlcv_df(n=100, uppercase=False)


# ---------------------------------------------------------------------------
# fetch-indicator CLI
# ---------------------------------------------------------------------------

class TestCLIIndicator:
    def _run_cli(self, args: list[str]):
        """Run cli.main() with given args."""
        from croesus_toolset.cli import main
        with patch("sys.argv", ["croesus"] + args):
            try:
                main()
                return 0, ""
            except SystemExit as e:
                return e.code if isinstance(e.code, int) else 1, ""

    @patch("croesus_toolset.cli._get_loader_and_df")
    def test_fetch_rsi_exits_0(self, mock_fetch):
        mock_fetch.return_value = _mock_loader_df()
        rc, _ = self._run_cli(["fetch-indicator", "--asset", "AAPL", "--indicator", "rsi_14", "--interval", "1d"])
        assert rc == 0

    @patch("croesus_toolset.cli._get_loader_and_df")
    def test_fetch_rsi_json_sane(self, mock_fetch, capsys):
        mock_fetch.return_value = _mock_loader_df()
        self._run_cli(["fetch-indicator", "--asset", "AAPL", "--indicator", "rsi_14"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["ok"] is True
        assert data["indicator"] == "rsi_14"
        assert isinstance(data["value"], float)
        assert 0 <= data["value"] <= 100

    @patch("croesus_toolset.cli._get_loader_and_df")
    def test_fetch_macd_json_sane(self, mock_fetch, capsys):
        mock_fetch.return_value = _mock_loader_df()
        self._run_cli(["fetch-indicator", "--asset", "AAPL", "--indicator", "macd"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["ok"] is True
        assert "macd_line" in data["value"]

    def test_fetch_unknown_indicator_exits_1(self, capsys):
        rc, _ = self._run_cli(["fetch-indicator", "--asset", "AAPL", "--indicator", "unknown_thing"])
        assert rc == 1
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["ok"] is False

    @patch("croesus_toolset.cli._get_loader_and_df")
    def test_fetch_no_data_exits_1(self, mock_fetch, capsys):
        mock_fetch.side_effect = ValueError("No data returned for INVALID")
        rc, _ = self._run_cli(["fetch-indicator", "--asset", "INVALID", "--indicator", "rsi_14"])
        assert rc == 1

    # --- Parameterised indicator tests ---

    @patch("croesus_toolset.cli._get_loader_and_df")
    def test_rsi_21_works(self, mock_fetch, capsys):
        """rsi_21 should be accepted (was rejected before)."""
        mock_fetch.return_value = _mock_loader_df()
        rc, _ = self._run_cli(["fetch-indicator", "--asset", "BTC-USD", "--indicator", "rsi_21"])
        assert rc == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["ok"] is True
        assert data["indicator"] == "rsi_21"
        assert isinstance(data["value"], float)

    @patch("croesus_toolset.cli._get_loader_and_df")
    def test_ema_50_works(self, mock_fetch, capsys):
        """ema_50 should be accepted (was rejected before)."""
        mock_fetch.return_value = _mock_loader_df()
        rc, _ = self._run_cli(["fetch-indicator", "--asset", "BTC-USD", "--indicator", "ema_50"])
        assert rc == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["ok"] is True
        assert data["indicator"] == "ema_50"

    @patch("croesus_toolset.cli._get_loader_and_df")
    def test_macd_12_26_works(self, mock_fetch, capsys):
        """macd_12_26_9 should be accepted."""
        mock_fetch.return_value = _mock_loader_df()
        rc, _ = self._run_cli(["fetch-indicator", "--asset", "BTC-USD", "--indicator", "macd_12_26_9"])
        assert rc == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["ok"] is True
        assert "macd_line" in data["value"]

    @patch("croesus_toolset.cli._get_loader_and_df")
    def test_sma_200_works(self, mock_fetch, capsys):
        mock_fetch.return_value = _mock_ohlcv_df(n=250)  # need 200+ rows for SMA 200
        rc, _ = self._run_cli(["fetch-indicator", "--asset", "SPY", "--indicator", "sma_200"])
        assert rc == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["ok"] is True
        assert isinstance(data["value"], float)

    # --- Loader selection tests ---

    @patch("croesus_toolset.cli._get_loader_and_df")
    def test_loader_flag_yfinance(self, mock_fetch):
        mock_fetch.return_value = _mock_loader_df()
        self._run_cli(["fetch-indicator", "--asset", "AAPL", "--indicator", "rsi_14", "--loader", "yfinance"])
        mock_fetch.assert_called_once()
        call_args = mock_fetch.call_args
        assert call_args[0][2] == "yfinance"

    @patch("croesus_toolset.cli._get_loader_and_df")
    def test_loader_flag_ccxt(self, mock_fetch):
        mock_fetch.return_value = _mock_loader_df()
        self._run_cli([
            "fetch-indicator", "--asset", "BTC/USDT", "--indicator", "rsi_14",
            "--loader", "ccxt", "--exchange", "binance",
        ])
        mock_fetch.assert_called_once()
        call_args = mock_fetch.call_args
        assert call_args[0][2] == "ccxt"

    def test_exit_code_2_on_missing_required_flag(self):
        rc, _ = self._run_cli(["fetch-indicator", "--indicator", "rsi_14"])
        assert rc == 2

    def test_exit_code_2_on_unknown_subcommand(self):
        rc, _ = self._run_cli(["bogus-command"])
        assert rc == 2


# ---------------------------------------------------------------------------
# fetch-factors CLI
# ---------------------------------------------------------------------------

class TestCLIFetchFactors:
    def _run_cli(self, args: list[str]):
        from croesus_toolset.cli import main
        with patch("sys.argv", ["croesus"] + args):
            try:
                main()
                return 0, ""
            except SystemExit as e:
                return e.code if isinstance(e.code, int) else 1, ""

    @patch("croesus_toolset.cli._get_loader_and_df")
    def test_fetch_factors_exits_0(self, mock_fetch):
        mock_fetch.return_value = _mock_loader_df()
        rc, _ = self._run_cli([
            "fetch-factors", "--asset", "BTC-USD",
            "--factors", "rsi_14,momentum_12_1",
        ])
        assert rc == 0

    @patch("croesus_toolset.cli._get_loader_and_df")
    def test_fetch_factors_json_structure(self, mock_fetch, capsys):
        mock_fetch.return_value = _mock_loader_df()
        self._run_cli([
            "fetch-factors", "--asset", "BTC-USD",
            "--factors", "rsi_14,volatility_30",
        ])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["ok"] is True
        assert data["symbol"] == "BTC-USD"
        assert "rsi_14" in data["factors"]
        assert "volatility_30" in data["factors"]
        assert isinstance(data["factors"]["rsi_14"], float)

    @patch("croesus_toolset.cli._get_loader_and_df")
    def test_fetch_factors_with_ccxt_loader(self, mock_fetch, capsys):
        mock_fetch.return_value = _mock_loader_df()
        self._run_cli([
            "fetch-factors", "--asset", "BTC/USDT",
            "--factors", "rsi_14",
            "--loader", "ccxt", "--exchange", "binance",
        ])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["ok"] is True
        assert data["loader"] == "ccxt"

    def test_fetch_factors_missing_asset_exits_2(self):
        rc, _ = self._run_cli(["fetch-factors", "--factors", "rsi_14"])
        assert rc == 2

    def test_fetch_factors_missing_factors_exits_2(self):
        rc, _ = self._run_cli(["fetch-factors", "--asset", "BTC-USD"])
        assert rc == 2

    @patch("croesus_toolset.cli._get_loader_and_df")
    def test_fetch_factors_unknown_factor_handled(self, mock_fetch, capsys):
        mock_fetch.return_value = _mock_loader_df()
        self._run_cli([
            "fetch-factors", "--asset", "BTC-USD",
            "--factors", "nonexistent_factor",
        ])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["ok"] is True
        assert "error" in data["factors"]["nonexistent_factor"]


# ---------------------------------------------------------------------------
# risk-xray CLI
# ---------------------------------------------------------------------------

class TestCLIRiskXray:
    def _run_cli(self, args: list[str]):
        from croesus_toolset.cli import main
        with patch("sys.argv", ["croesus"] + args):
            try:
                main()
                return 0
            except SystemExit as e:
                return e.code if isinstance(e.code, int) else 1

    def test_risk_xray_missing_file_exits_1(self, capsys):
        rc = self._run_cli(["risk-xray", "--portfolio", "/nonexistent.json"])
        assert rc == 1
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["ok"] is False

    def test_risk_xray_bad_json_exits_1(self, tmp_path, capsys):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json {{{")
        rc = self._run_cli(["risk-xray", "--portfolio", str(bad_file)])
        assert rc == 1

    def test_risk_xray_no_weights_exits_1(self, tmp_path, capsys):
        pf = tmp_path / "holdings.json"
        pf.write_text(json.dumps({"symbols": ["AAPL"]}))
        rc = self._run_cli(["risk-xray", "--portfolio", str(pf)])
        assert rc == 1

    @patch("croesus_toolset.cli._get_loader_and_df")
    def test_risk_xray_exits_0_with_data(self, mock_fetch, tmp_path, capsys):
        rng = np.random.default_rng(42)
        n = 100
        dates = pd.bdate_range("2025-01-01", periods=n)
        mock_fetch.return_value = pd.DataFrame({
            "close": 100.0 + np.cumsum(rng.normal(0, 1, size=n)),
        }, index=dates)

        pf = tmp_path / "holdings.json"
        pf.write_text(json.dumps({
            "symbols": ["AAPL", "MSFT"],
            "weights": {"AAPL": 0.6, "MSFT": 0.4},
        }))
        rc = self._run_cli(["risk-xray", "--portfolio", str(pf)])
        assert rc == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["ok"] if "ok" in data else "concentration" in data
