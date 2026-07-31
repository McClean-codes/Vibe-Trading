"""croesus-toolset: enrichment toolset lifted from Vibe-Trading.

Public surface:
- indicators: compute_rsi, compute_macd, compute_bollinger, compute_sma, compute_ema
- backtest: bar_returns, sharpe_ratio, max_drawdown
- risk: compute_risk_xray, render_risk_xray_markdown
"""

__version__ = "0.1.0"

from croesus_toolset.indicators import (
    compute_bollinger,
    compute_ema,
    compute_macd,
    compute_rsi,
    compute_sma,
)
from croesus_toolset.backtest import bar_returns, max_drawdown, sharpe_ratio
from croesus_toolset.risk import compute_risk_xray, render_risk_xray_markdown

__all__ = [
    "compute_bollinger",
    "compute_ema",
    "compute_macd",
    "compute_rsi",
    "compute_sma",
    "bar_returns",
    "max_drawdown",
    "sharpe_ratio",
    "compute_risk_xray",
    "render_risk_xray_markdown",
]
