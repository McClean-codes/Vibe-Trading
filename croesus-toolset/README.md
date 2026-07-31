# croesus-toolset

Enrichment toolset lifted from [Vibe-Trading](https://github.com/HKUDS/Vibe-Trading) (MIT License).

Provides clean, standalone Python APIs for:
- **Technical indicators** — RSI, MACD, Bollinger Bands, SMA, EMA
- **Backtest metrics** — bar returns, Sharpe ratio, max drawdown
- **Portfolio risk x-ray** — concentration (HHI), volatility, drawdown, VaR/ES, diversification

## Installation

```bash
pip install -e .
# With data-fetch support:
pip install -e ".[fetch]"
```

## Python API

```python
import pandas as pd
from croesus_toolset import compute_rsi, compute_macd, compute_bollinger
from croesus_toolset import sharpe_ratio, max_drawdown
from croesus_toolset import compute_risk_xray, render_risk_xray_markdown

# Indicators
close = pd.Series([...])  # your close prices
rsi = compute_rsi(close, period=14)
macd = compute_macd(close)  # returns dict with macd_line, signal_line, histogram
bb = compute_bollinger(close)  # returns dict with upper, middle, lower

# Backtest metrics
returns = close.pct_change().dropna()
sharpe = sharpe_ratio(returns)
mdd = max_drawdown(close)

# Risk x-ray
closes_panel = pd.DataFrame({"AAPL": [...], "MSFT": [...]})
weights = {"AAPL": 0.6, "MSFT": 0.4}
report = compute_risk_xray(closes_panel, weights)
print(render_risk_xray_markdown(report))
```

## CLI

```bash
# Fetch indicator for a symbol
croesus fetch-indicator --asset BTC-USD --indicator rsi_14 --interval 1d

# Portfolio risk x-ray
croesus risk-xray --portfolio ./holdings.json

# Risk x-ray with artifact output
croesus risk-xray --portfolio ./holdings.json --out-dir ./croesus_runs/run-001/
```

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | Success (`{"ok": true}` in JSON output) |
| `1` | Data/logic error (`{"ok": false}` — no data, bad portfolio, etc.) |
| `2` | Argument error (missing flag, unknown subcommand — argparse default) |

This makes the CLI cron-safe: a cron job that treats non-zero exit as an alert
will fire correctly on data-fetch failures (exit 1) and argument mistakes (exit 2),
while a clean run exits 0.

### holdings.json format

**Canonical form** (recommended):

```json
{
  "symbols": ["AAPL", "MSFT", "GOOG"],
  "weights": {"AAPL": 0.4, "MSFT": 0.3, "GOOG": 0.3}
}
```

**Alias form** (also accepted):

```json
{
  "holdings": [
    {"asset": "AAPL", "weight": 0.4},
    {"asset": "MSFT", "weight": 0.3},
    {"asset": "GOOG", "weight": 0.3}
  ]
}
```

Both formats produce identical risk x-ray output. The `weights` key is the
canonical form; `holdings` is an alias that translates internally.

### risk-xray artifact emitter

When `--out-dir <path>` is specified, `risk-xray` writes two files alongside
the JSON printed to stdout:

| File | Contents |
|------|----------|
| `risk_xray.json` | Full machine-readable risk report |
| `risk_xray.md` | Human-friendly summary (concentration, volatility, drawdown, tail risk, diversification) |

Both files land in the same run directory, matching the artifact pair that
Vibe-Trading produces during portfolio backtests.

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## What was lifted

| Source | Destination | Description |
|--------|-------------|-------------|
| `agent/src/tools/technical_indicator_tool.py` | `src/croesus_toolset/indicators.py` | RSI, MACD, Bollinger, SMA, EMA (pure math) |
| `agent/backtest/metrics.py` | `src/croesus_toolset/backtest.py` | Bar returns, Sharpe, max drawdown |
| `agent/backtest/risk_xray.py` | `src/croesus_toolset/risk.py` | Portfolio risk x-ray (concentration, vol, drawdown, tail, diversification) |

## What was NOT lifted

- `agent/src/cli/`, `agent/api_server.py`, `agent/mcp_server.py` — agent runtime
- `agent/src/swarm/`, `agent/src/chat/` — LLM loop / adapters
- `agent/src/factors/zoo/` — Alpha Zoo (452 factors, not yet included)
- `agent/backtest/loaders/` — data loader pipeline (yfinance, akshare, etc.)
- `agent/backtest/engines/` — backtest execution engines
- `frontend/`, `Dockerfile`, `requirements-lock.txt`

## License

MIT — same as Vibe-Trading. See [LICENSE](./LICENSE).
