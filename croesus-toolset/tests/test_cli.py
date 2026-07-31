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

    # --- exit-code 0 on ok:true, exit-code 1 on ok:false ---

    @patch("croesus_toolset.cli._fetch_ohlcv_yfinance")
    def test_exit_code_0_on_ok_true(self, mock_fetch):
        """Process must exit 0 when JSON body has ok=true."""
        mock_fetch.return_value = self._mock_yfinance_df()
        rc, _ = self._run_cli(["fetch-indicator", "--asset", "BTC-USD", "--indicator", "rsi_14", "--interval", "1d"])
        assert rc == 0

    @patch("croesus_toolset.cli._fetch_ohlcv_yfinance")
    def test_exit_code_1_on_ok_false(self, mock_fetch):
        """Process must exit 1 when JSON body has ok=false (e.g. no data)."""
        mock_fetch.return_value = None
        rc, _ = self._run_cli(["fetch-indicator", "--asset", "NONEXISTENT", "--indicator", "rsi_14", "--interval", "1h"])
        assert rc == 1

    def test_exit_code_2_on_missing_required_flag(self):
        """Process must exit 2 on argument/config errors (missing --asset)."""
        rc, _ = self._run_cli(["fetch-indicator", "--indicator", "rsi_14"])
        assert rc == 2

    def test_exit_code_2_on_unknown_subcommand(self):
        """Process must exit 2 on unknown subcommand."""
        rc, _ = self._run_cli(["bogus-command"])
        assert rc == 2


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
        assert "concentration" in data

    # --- holdings alias ---

    @patch("croesus_toolset.cli._fetch_ohlcv_yfinance")
    def test_risk_xray_accepts_holdings_alias(self, mock_fetch, tmp_path, capsys):
        """risk-xray should accept {holdings: [{asset, weight}, ...]} format."""
        import numpy as np
        rng = np.random.default_rng(42)
        n = 100
        dates = pd.bdate_range("2025-01-01", periods=n)
        mock_fetch.return_value = pd.DataFrame({
            "Close": 100.0 + np.cumsum(rng.normal(0, 1, size=n)),
        }, index=dates)

        pf = tmp_path / "holdings.json"
        pf.write_text(json.dumps({
            "holdings": [
                {"asset": "AAPL", "weight": 0.6},
                {"asset": "MSFT", "weight": 0.4},
            ],
        }))
        rc = self._run_cli(["risk-xray", "--portfolio", str(pf)])
        assert rc == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "concentration" in data
        assert "AAPL" in data["inputs"]["weights"]
        assert "MSFT" in data["inputs"]["weights"]

    @patch("croesus_toolset.cli._fetch_ohlcv_yfinance")
    def test_risk_xray_holdings_matches_weights_output(self, mock_fetch, tmp_path):
        """holdings alias should produce identical risk output to weights form."""
        import numpy as np
        rng = np.random.default_rng(42)
        n = 100
        dates = pd.bdate_range("2025-01-01", periods=n)
        mock_fetch.return_value = pd.DataFrame({
            "Close": 100.0 + np.cumsum(rng.normal(0, 1, size=n)),
        }, index=dates)

        pf_weights = tmp_path / "weights.json"
        pf_weights.write_text(json.dumps({
            "symbols": ["AAPL", "MSFT"],
            "weights": {"AAPL": 0.6, "MSFT": 0.4},
        }))
        rc1 = self._run_cli(["risk-xray", "--portfolio", str(pf_weights)])

        pf_holdings = tmp_path / "holdings.json"
        pf_holdings.write_text(json.dumps({
            "holdings": [
                {"asset": "AAPL", "weight": 0.6},
                {"asset": "MSFT", "weight": 0.4},
            ],
        }))
        rc2 = self._run_cli(["risk-xray", "--portfolio", str(pf_holdings)])

        assert rc1 == 0
        assert rc2 == 0

    # --- --out-dir dual artifact emitter ---

    @patch("croesus_toolset.cli._fetch_ohlcv_yfinance")
    def test_risk_xray_out_dir_creates_both_files(self, mock_fetch, tmp_path):
        """--out-dir should create both risk_xray.json and risk_xray.md."""
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

        out_dir = tmp_path / "xray_output"
        rc = self._run_cli(["risk-xray", "--portfolio", str(pf), "--out-dir", str(out_dir)])
        assert rc == 0
        assert (out_dir / "risk_xray.json").exists()
        assert (out_dir / "risk_xray.md").exists()

    @patch("croesus_toolset.cli._fetch_ohlcv_yfinance")
    def test_risk_xray_out_dir_json_valid(self, mock_fetch, tmp_path):
        """risk_xray.json should be valid JSON matching stdout output."""
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

        out_dir = tmp_path / "xray_output"
        self._run_cli(["risk-xray", "--portfolio", str(pf), "--out-dir", str(out_dir)])

        data = json.loads((out_dir / "risk_xray.json").read_text())
        assert "concentration" in data
        assert "volatility" in data
        assert "drawdown" in data

    @patch("croesus_toolset.cli._fetch_ohlcv_yfinance")
    def test_risk_xray_out_dir_md_content(self, mock_fetch, tmp_path):
        """risk_xray.md should contain human-friendly summary."""
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

        out_dir = tmp_path / "xray_output"
        self._run_cli(["risk-xray", "--portfolio", str(pf), "--out-dir", str(out_dir)])

        md_content = (out_dir / "risk_xray.md").read_text()
        assert "Risk" in md_content or "risk" in md_content.lower()
        assert "Drawdown" in md_content or "drawdown" in md_content.lower()
        assert "Volatility" in md_content or "volatility" in md_content.lower()
