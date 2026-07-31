"""Tests for croesus_toolset.cli — CLI entrypoint."""

import json
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# fetch-indicator CLI
# ---------------------------------------------------------------------------

class TestCLIIndicator:
    def _run_cli(self, args: list[str]):
        """Run cli.main() with given args and capture output."""
        from croesus_toolset.cli import main
        with patch("sys.argv", ["croesus"] + args):
            # main() calls sys.exit, so we catch SystemExit
            try:
                main()
                return 0, ""
            except SystemExit as e:
                return e.code if isinstance(e.code, int) else 1, ""

    def _mock_yfinance_df(self, n=100):
        """Create a mock yfinance DataFrame."""
        import numpy as np
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2025-01-01", periods=n)
        closes = 100.0 + np.cumsum(rng.normal(0, 1, size=n))
        df = pd.DataFrame({
            "Open": closes - 1,
            "High": closes + 2,
            "Low": closes - 2,
            "Close": closes,
            "Volume": rng.integers(1000, 10000, size=n),
        }, index=dates)
        return df

    @patch("croesus_toolset.cli._fetch_ohlcv_yfinance")
    def test_fetch_rsi_exits_0(self, mock_fetch):
        mock_fetch.return_value = self._mock_yfinance_df()
        rc, _ = self._run_cli(["fetch-indicator", "--asset", "AAPL", "--indicator", "rsi_14", "--interval", "1d"])
        assert rc == 0

    @patch("croesus_toolset.cli._fetch_ohlcv_yfinance")
    def test_fetch_rsi_json_sane(self, mock_fetch, capsys):
        mock_fetch.return_value = self._mock_yfinance_df()
        self._run_cli(["fetch-indicator", "--asset", "AAPL", "--indicator", "rsi_14"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["ok"] is True
        assert data["indicator"] == "rsi_14"
        assert isinstance(data["value"], float)
        assert 0 <= data["value"] <= 100

    @patch("croesus_toolset.cli._fetch_ohlcv_yfinance")
    def test_fetch_macd_json_sane(self, mock_fetch, capsys):
        mock_fetch.return_value = self._mock_yfinance_df()
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

    @patch("croesus_toolset.cli._fetch_ohlcv_yfinance")
    def test_fetch_no_data_exits_1(self, mock_fetch, capsys):
        mock_fetch.return_value = pd.DataFrame()
        rc, _ = self._run_cli(["fetch-indicator", "--asset", "INVALID", "--indicator", "rsi_14"])
        assert rc == 1


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

    @patch("croesus_toolset.cli._fetch_ohlcv_yfinance")
    def test_risk_xray_exits_0_with_data(self, mock_fetch, tmp_path, capsys):
        import numpy as np
        rng = np.random.default_rng(42)
        n = 100
        dates = pd.bdate_range("2025-01-01", periods=n)
        mock_fetch.return_value = pd.DataFrame({
            "Close": 100.0 + np.cumsum(rng.normal(0, 1, size=n)),
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
