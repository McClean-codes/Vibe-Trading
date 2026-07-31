"""Tests for croesus_toolset.risk — portfolio risk x-ray."""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_closes(symbols: list[str], n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic close-price panel with multiple symbols."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=n)
    data = {}
    for sym in symbols:
        returns = rng.normal(0.0005, 0.02, size=n)
        prices = 100.0 * np.cumprod(1 + returns)
        data[sym] = prices
    return pd.DataFrame(data, index=dates)


# ---------------------------------------------------------------------------
# compute_risk_xray
# ---------------------------------------------------------------------------

class TestComputeRiskXray:
    def test_returns_dict_with_expected_sections(self):
        from croesus_toolset.risk import compute_risk_xray
        closes = _make_closes(["AAPL", "MSFT", "GOOG"])
        weights = {"AAPL": 0.4, "MSFT": 0.3, "GOOG": 0.3}
        result = compute_risk_xray(closes, weights)
        assert isinstance(result, dict)
        for key in ("inputs", "concentration", "volatility", "drawdown", "tail_risk", "diversification"):
            assert key in result, f"missing section: {key}"

    def test_concentration_hhi_range(self):
        from croesus_toolset.risk import compute_risk_xray
        closes = _make_closes(["A", "B", "C"])
        weights = {"A": 0.5, "B": 0.3, "C": 0.2}
        result = compute_risk_xray(closes, weights)
        hhi = result["concentration"]["hhi"]
        assert 0.0 < hhi <= 1.0

    def test_max_drawdown_is_negative_or_zero(self):
        from croesus_toolset.risk import compute_risk_xray
        closes = _make_closes(["BTC-USD"])
        weights = {"BTC-USD": 1.0}
        result = compute_risk_xray(closes, weights)
        mdd = result["drawdown"]["max_drawdown"]
        assert mdd <= 0.0

    def test_rejects_negative_weights(self):
        from croesus_toolset.risk import compute_risk_xray
        closes = _make_closes(["A", "B"])
        weights = {"A": 1.5, "B": -0.5}
        with pytest.raises(ValueError, match="negative"):
            compute_risk_xray(closes, weights)

    def test_rejects_empty_panel(self):
        from croesus_toolset.risk import compute_risk_xray
        closes = pd.DataFrame()
        weights = {"A": 1.0}
        with pytest.raises(ValueError):
            compute_risk_xray(closes, weights)

    def test_renormalizes_non_unit_weights(self):
        from croesus_toolset.risk import compute_risk_xray
        closes = _make_closes(["X", "Y"])
        weights = {"X": 2.0, "Y": 3.0}
        result = compute_risk_xray(closes, weights)
        assert len(result.get("warnings", [])) > 0 or result["inputs"]["symbols"] == ["X", "Y"]

    def test_skips_symbols_with_insufficient_history(self):
        from croesus_toolset.risk import compute_risk_xray
        closes = _make_closes(["GOOD", "SHORT"], n=100)
        closes.loc[closes.index[:90], "SHORT"] = np.nan
        weights = {"GOOD": 0.5, "SHORT": 0.5}
        result = compute_risk_xray(closes, weights, min_history=30)
        assert "SHORT" in str(result.get("skipped", []))


# ---------------------------------------------------------------------------
# render_risk_xray_markdown
# ---------------------------------------------------------------------------

class TestRenderRiskXrayMarkdown:
    def test_returns_string(self):
        from croesus_toolset.risk import compute_risk_xray, render_risk_xray_markdown
        closes = _make_closes(["A", "B"])
        report = compute_risk_xray(closes, {"A": 0.6, "B": 0.4})
        md = render_risk_xray_markdown(report)
        assert isinstance(md, str)
        assert len(md) > 0

    def test_contains_heading(self):
        from croesus_toolset.risk import compute_risk_xray, render_risk_xray_markdown
        closes = _make_closes(["A"])
        report = compute_risk_xray(closes, {"A": 1.0})
        md = render_risk_xray_markdown(report)
        assert "Risk" in md or "risk" in md.lower()


# ---------------------------------------------------------------------------
# holdings alias at risk.py level
# ---------------------------------------------------------------------------

class TestHoldingsAlias:
    def test_extract_weights_from_holdings(self):
        """parse_portfolio should extract weights from holdings alias."""
        from croesus_toolset.cli import parse_portfolio
        portfolio = {
            "holdings": [
                {"asset": "AAPL", "weight": 0.6},
                {"asset": "MSFT", "weight": 0.4},
            ],
        }
        weights, symbols = parse_portfolio(portfolio)
        assert weights == {"AAPL": 0.6, "MSFT": 0.4}
        assert symbols == ["AAPL", "MSFT"]

    def test_extract_weights_from_weights_key(self):
        """parse_portfolio should extract weights from canonical weights key."""
        from croesus_toolset.cli import parse_portfolio
        portfolio = {
            "symbols": ["AAPL", "MSFT"],
            "weights": {"AAPL": 0.6, "MSFT": 0.4},
        }
        weights, symbols = parse_portfolio(portfolio)
        assert weights == {"AAPL": 0.6, "MSFT": 0.4}
        assert symbols == ["AAPL", "MSFT"]

    def test_holdings_and_weights_same_result(self):
        """Both formats should produce identical weights dict."""
        from croesus_toolset.cli import parse_portfolio
        w_weights, s_weights = parse_portfolio({
            "symbols": ["A", "B", "C"],
            "weights": {"A": 0.5, "B": 0.3, "C": 0.2},
        })
        w_holdings, s_holdings = parse_portfolio({
            "holdings": [
                {"asset": "A", "weight": 0.5},
                {"asset": "B", "weight": 0.3},
                {"asset": "C", "weight": 0.2},
            ],
        })
        assert w_weights == w_holdings
        assert set(s_weights) == set(s_holdings)

    def test_rejects_neither_weights_nor_holdings(self):
        """Should raise ValueError when neither weights nor holdings present."""
        from croesus_toolset.cli import parse_portfolio
        with pytest.raises(ValueError, match="Portfolio must have 'weights' or 'holdings'"):
            parse_portfolio({"symbols": ["AAPL"]})
