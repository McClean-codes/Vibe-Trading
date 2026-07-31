"""CLI entrypoint for croesus-toolset.

Commands:
    fetch-indicator  — fetch OHLCV data and compute technical indicators
    fetch-factors    — fetch OHLCV data and compute alpha-zoo factors
    risk-xray        — compute portfolio risk x-ray from a holdings JSON file

Usage:
    croesus fetch-indicator --asset BTCUSDT --indicator rsi_14 --interval 1h
    croesus fetch-indicator --indicator rsi_21 --asset BTC-USD --loader ccxt --exchange binance
    croesus fetch-factors --asset BTC-USD --factors rsi_14,momentum_12_1 --window 90d
    croesus risk-xray --portfolio ./holdings.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Portfolio parsing — canonical weights + holdings alias
# ---------------------------------------------------------------------------

def parse_portfolio(portfolio: dict[str, Any]) -> tuple[dict[str, float], list[str]]:
    """Extract weights and symbols from a portfolio dict.

    Accepts two formats:
      Canonical:  {"symbols": [...], "weights": {"SYM": w, ...}}
      Alias:      {"holdings": [{"asset": "SYM", "weight": w}, ...]}

    Returns:
        (weights, symbols) tuple.

    Raises:
        ValueError: if neither 'weights' nor 'holdings' key is present.
    """
    weights = portfolio.get("weights")
    symbols = portfolio.get("symbols")

    if weights is not None:
        if symbols is None:
            symbols = list(weights.keys())
        return dict(weights), list(symbols)

    holdings = portfolio.get("holdings")
    if holdings is not None:
        extracted: dict[str, float] = {}
        extracted_symbols: list[str] = []
        for item in holdings:
            asset = item["asset"]
            extracted[asset] = float(item["weight"])
            extracted_symbols.append(asset)
        return extracted, extracted_symbols

    raise ValueError("Portfolio must have 'weights' or 'holdings' key")


# ---------------------------------------------------------------------------
# Indicator name parsing — supports both static keys and parameterised names
# ---------------------------------------------------------------------------

_INDICATOR_ALIASES: dict[str, str] = {
    "rsi": "rsi",
    "macd": "macd",
    "bollinger": "bollinger",
    "sma": "sma",
    "ema": "ema",
}

# Pattern: indicator_name followed by optional _period (e.g. rsi_14, ema_50)
_PARAM_RE = re.compile(r"^([a-z]+)(?:_(\d+))?$", re.IGNORECASE)


def _parse_indicator(name: str):
    """Parse indicator name into (base_name, params).

    Returns (base_name, params_dict) or raises ValueError.
    """
    # Special-case MACD: macd_12_26_9 → fast=12, slow=26, signal=9
    if name.lower().startswith("macd"):
        parts = name.split("_")
        if len(parts) == 1:
            return "macd", {}
        elif len(parts) == 4:
            return "macd", {
                "fast": int(parts[1]),
                "slow": int(parts[2]),
                "signal": int(parts[3]),
            }
        else:
            raise ValueError(
                f"MACD format: macd or macd_fast_slow_signal, got {name!r}"
            )

    m = _PARAM_RE.match(name)
    if not m:
        raise ValueError(f"Invalid indicator name: {name!r}")

    base = m.group(1).lower()
    period_str = m.group(2)

    if base in ("rsi", "sma", "ema"):
        period = int(period_str) if period_str else 14
        return base, {"period": period}
    elif base == "bollinger":
        period = int(period_str) if period_str else 20
        return base, {"period": period}
    else:
        raise ValueError(f"Unknown indicator: {name!r}")


def _compute_indicator(base_name: str, params: dict, close):
    """Compute an indicator given base name and params."""
    from croesus_toolset.indicators import (
        compute_bollinger, compute_ema, compute_macd, compute_rsi, compute_sma,
    )

    if base_name == "rsi":
        return compute_rsi(close, period=params.get("period", 14))
    elif base_name == "macd":
        return compute_macd(
            close,
            fast=params.get("fast", 12),
            slow=params.get("slow", 26),
            signal=params.get("signal", 9),
        )
    elif base_name == "bollinger":
        return compute_bollinger(close, period=params.get("period", 20))
    elif base_name == "sma":
        return compute_sma(close, period=params.get("period", 20))
    elif base_name == "ema":
        return compute_ema(close, period=params.get("period", 20))
    else:
        raise ValueError(f"Unknown indicator: {base_name!r}")


# ---------------------------------------------------------------------------
# Loader dispatch
# ---------------------------------------------------------------------------

def _get_loader_and_df(symbol: str, interval: str, loader_name: str, **kwargs):
    """Fetch data via the named loader, return df."""
    from croesus_toolset.loaders import get_loader

    loader = get_loader(loader_name)
    window = kwargs.pop("window", "90d")
    df = loader.fetch_ohlcv(symbol, interval=interval, window=window, **kwargs)
    return df


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_fetch_indicator(args):
    """Fetch OHLCV and compute technical indicators."""
    symbol = args.asset
    indicator = args.indicator
    interval = args.interval
    loader_name = args.loader

    try:
        base_name, params = _parse_indicator(indicator)
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1

    try:
        df = _get_loader_and_df(
            symbol, interval, loader_name, exchange=args.exchange
        )
    except (ImportError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1

    # Get close prices
    if "close" in df.columns:
        close = df["close"]
    elif "Close" in df.columns:
        close = df["Close"]
    else:
        print(json.dumps({"ok": False, "error": "No close price column in data"}))
        return 1

    # Compute indicator
    try:
        result = _compute_indicator(base_name, params, close)
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1

    latest_close = float(close.iloc[-1]) if len(close) > 0 else None
    latest_date = str(close.index[-1])[:10] if hasattr(close, "index") and len(close) > 0 else None

    output = {
        "ok": True,
        "symbol": symbol,
        "interval": interval,
        "indicator": indicator,
        "value": result,
        "latest_close": latest_close,
        "latest_date": latest_date,
    }
    print(json.dumps(output, ensure_ascii=False, default=str))
    return 0


def cmd_fetch_factors(args):
    """Fetch OHLCV and compute alpha-zoo factors."""
    from croesus_toolset.factors import get_factor

    symbol = args.asset
    interval = args.interval
    loader_name = args.loader
    factor_names = [f.strip() for f in args.factors.split(",")]

    try:
        df = _get_loader_and_df(
            symbol, interval, loader_name,
            exchange=args.exchange, window=args.window,
        )
    except (ImportError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1

    results = {}
    for name in factor_names:
        try:
            func = get_factor(name)
            series = func(df)
            # Take the last non-NaN value
            last_val = series.dropna().iloc[-1] if not series.dropna().empty else None
            results[name] = last_val
        except (KeyError, Exception) as exc:
            results[name] = {"error": str(exc)}

    output = {
        "ok": True,
        "symbol": symbol,
        "interval": interval,
        "window": args.window,
        "loader": loader_name,
        "factors": results,
    }
    print(json.dumps(output, ensure_ascii=False, default=str))
    return 0


def cmd_risk_xray(args):
    """Compute portfolio risk x-ray from a holdings JSON file."""
    from croesus_toolset.risk import compute_risk_xray, render_risk_xray_markdown

    portfolio_path = Path(args.portfolio)
    if not portfolio_path.exists():
        print(json.dumps({"ok": False, "error": f"Portfolio file not found: {portfolio_path}"}))
        return 1

    try:
        portfolio = json.loads(portfolio_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(json.dumps({"ok": False, "error": f"Failed to parse portfolio: {exc}"}))
        return 1

    try:
        weights, symbols = parse_portfolio(portfolio)
    except (ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1

    if not weights:
        print(json.dumps({"ok": False, "error": "Portfolio must have 'weights' or 'holdings' key"}))
        return 1

    # Fetch close prices for each symbol
    import pandas as pd
    loader_name = args.loader
    closes_dict = {}
    for sym in symbols:
        try:
            df = _get_loader_and_df(sym, args.interval, loader_name, exchange=args.exchange)
            if "close" in df.columns:
                closes_dict[sym] = df["close"].reset_index(drop=True)
            elif "Close" in df.columns:
                closes_dict[sym] = df["Close"].reset_index(drop=True)
        except (ImportError, ValueError):
            continue

    if not closes_dict:
        print(json.dumps({"ok": False, "error": "No price data fetched for any symbol"}))
        return 1

    closes = pd.DataFrame(closes_dict)

    try:
        report = compute_risk_xray(closes, weights)
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1

    # Dual artifact emitter: write JSON + MD to --out-dir if specified
    out_dir = getattr(args, "out_dir", None)
    if out_dir:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        json_file = out_path / "risk_xray.json"
        json_file.write_text(json.dumps(report, ensure_ascii=False, default=str, indent=2), encoding="utf-8")

        md_content = render_risk_xray_markdown(report)
        md_file = out_path / "risk_xray.md"
        md_file.write_text(md_content, encoding="utf-8")

    if args.format == "markdown":
        print(render_risk_xray_markdown(report))
    else:
        print(json.dumps(report, ensure_ascii=False, default=str))
    return 0


# ---------------------------------------------------------------------------
# NEW: gap-safe-returns, benchmark-returns, liquidation-price commands
# ---------------------------------------------------------------------------


def _parse_window(window: str) -> int:
    """Parse a window string like '30d', '7d', '90d' to days."""
    window = window.strip().lower()
    if window.endswith("d"):
        return int(window[:-1])
    if window.endswith("w"):
        return int(window[:-1]) * 7
    if window.endswith("m"):
        return int(window[:-1]) * 30
    return int(window)


def _fetch_close_prices(symbol: str, interval: str, days: int = 30, loader_name: str = "yfinance", exchange: str | None = None):
    """Fetch close prices for a symbol. Returns pd.Series or None."""
    try:
        df = _get_loader_and_df(symbol, interval, loader_name, exchange=exchange)
        if "close" in df.columns:
            return df["close"].reset_index(drop=True)
        elif "Close" in df.columns:
            return df["Close"].reset_index(drop=True)
    except (ImportError, ValueError):
        pass
    return None


def cmd_gap_safe_returns(args):
    """Compute gap-safe bar returns for an asset."""
    from croesus_toolset.backtest import gap_safe_bar_returns

    symbol = args.asset
    interval = args.interval
    window = _parse_window(args.window)
    loader_name = getattr(args, "loader", "yfinance")
    exchange = getattr(args, "exchange", None)

    close = _fetch_close_prices(symbol, interval, days=window, loader_name=loader_name, exchange=exchange)
    if close is None or len(close) < 2:
        print(json.dumps({"ok": False, "error": f"Insufficient data for {symbol}"}))
        return 1

    result = gap_safe_bar_returns(close, halted_threshold=args.threshold)
    halted_mask = result.attrs.get("halted_mask", None)

    output = {
        "ok": True,
        "symbol": symbol,
        "interval": interval,
        "window": args.window,
        "halted_threshold": args.threshold,
        "bars": len(result),
        "returns": result.tolist(),
        "halted_bars": int(halted_mask.sum()) if halted_mask is not None else 0,
        "mean_return": float(result.mean()),
        "std_return": float(result.std()),
    }
    print(json.dumps(output, ensure_ascii=False, default=str))
    return 0


def cmd_benchmark_returns(args):
    """Compute sign-safe benchmark returns for an asset."""
    from croesus_toolset.backtest import sign_safe_benchmark

    symbol = args.asset
    interval = args.interval
    window = _parse_window(args.window)
    loader_name = getattr(args, "loader", "yfinance")
    exchange = getattr(args, "exchange", None)

    close = _fetch_close_prices(symbol, interval, days=window, loader_name=loader_name, exchange=exchange)
    if close is None or len(close) < 2:
        print(json.dumps({"ok": False, "error": f"Insufficient data for {symbol}"}))
        return 1

    result = sign_safe_benchmark(close)
    total_return = float((1 + result).prod() - 1)

    output = {
        "ok": True,
        "symbol": symbol,
        "interval": interval,
        "window": args.window,
        "bars": len(result),
        "total_return": total_return,
        "mean_return": float(result.mean()),
        "std_return": float(result.std()),
    }
    print(json.dumps(output, ensure_ascii=False, default=str))
    return 0


def cmd_liquidation_price(args):
    """Compute USD-M perpetual liquidation price."""
    from croesus_toolset.backtest import usdm_liquidation_price

    try:
        liq = usdm_liquidation_price(
            entry_price=args.entry,
            leverage=args.leverage,
            side=args.side,
            margin_type=args.margin,
        )
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1

    output = {
        "ok": True,
        "entry_price": args.entry,
        "leverage": args.leverage,
        "side": args.side,
        "margin_type": args.margin,
        "liquidation_price": liq,
        "distance_pct": abs(liq - args.entry) / args.entry * 100,
    }
    print(json.dumps(output, ensure_ascii=False, default=str))
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="croesus",
        description="croesus-toolset: enrichment toolset lifted from Vibe-Trading",
    )
    sub = parser.add_subparsers(dest="command")

    # fetch-indicator
    fi = sub.add_parser("fetch-indicator", help="Fetch OHLCV and compute technical indicators")
    fi.add_argument("--asset", required=True, help="Trading symbol (e.g. BTCUSDT, AAPL)")
    fi.add_argument(
        "--indicator", required=True,
        help="Indicator: rsi_14, macd, bollinger, sma_20, ema_50, etc. "
             "Period is appended with _N (default 14 for RSI, 20 for SMA/EMA/Bollinger).",
    )
    fi.add_argument("--interval", default="1d", help="Bar interval (1h, 1d, etc.)")
    fi.add_argument(
        "--loader", default="yfinance", choices=["yfinance", "ccxt"],
        help="Data source loader (default: yfinance)",
    )
    fi.add_argument("--exchange", default=None, help="Exchange for ccxt loader (e.g. binance)")

    # fetch-factors
    ff = sub.add_parser("fetch-factors", help="Fetch OHLCV and compute alpha-zoo factors")
    ff.add_argument("--asset", required=True, help="Trading symbol")
    ff.add_argument(
        "--factors", required=True,
        help="Comma-separated factor names (e.g. rsi_14,momentum_12_1,volatility_30)",
    )
    ff.add_argument("--window", default="90d", help="How far back to fetch (default: 90d)")
    ff.add_argument("--interval", default="1d", help="Bar interval (1h, 1d, etc.)")
    ff.add_argument(
        "--loader", default="yfinance", choices=["yfinance", "ccxt"],
        help="Data source loader (default: yfinance)",
    )
    ff.add_argument("--exchange", default=None, help="Exchange for ccxt loader")

    # risk-xray
    rx = sub.add_parser("risk-xray", help="Compute portfolio risk x-ray")
    rx.add_argument("--portfolio", required=True, help="Path to holdings JSON file")
    rx.add_argument("--interval", default="1d", help="Bar interval for price data")
    rx.add_argument("--format", choices=["json", "markdown"], default="json", help="Output format")
    rx.add_argument(
        "--loader", default="yfinance", choices=["yfinance", "ccxt"],
        help="Data source loader (default: yfinance)",
    )
    rx.add_argument("--exchange", default=None, help="Exchange for ccxt loader")
    rx.add_argument("--out-dir", default=None, help="Write risk_xray.json and risk_xray.md to this directory")

    # gap-safe-returns
    gsr = sub.add_parser("gap-safe-returns", help="Compute gap-safe bar returns for an asset")
    gsr.add_argument("--asset", required=True, help="Trading symbol (e.g. BTC-USD)")
    gsr.add_argument("--interval", default="1h", help="Bar interval (1h, 1d, etc.)")
    gsr.add_argument("--window", default="30d", help="Data window (e.g. 30d, 90d)")
    gsr.add_argument("--threshold", type=int, default=5, help="Halted threshold (default: 5)")

    # benchmark-returns
    br = sub.add_parser("benchmark-returns", help="Compute sign-safe benchmark returns for an asset")
    br.add_argument("--asset", required=True, help="Trading symbol (e.g. BTC-USD)")
    br.add_argument("--interval", default="1d", help="Bar interval (1h, 1d, etc.)")
    br.add_argument("--window", default="90d", help="Data window (e.g. 30d, 90d)")

    # liquidation-price
    lp = sub.add_parser("liquidation-price", help="Compute USD-M perpetual liquidation price")
    lp.add_argument("--entry", type=float, required=True, help="Entry price (e.g. 60000)")
    lp.add_argument("--leverage", type=int, required=True, help="Leverage (e.g. 10)")
    lp.add_argument("--side", choices=["long", "short"], required=True, help="Position side")
    lp.add_argument("--margin", choices=["isolated", "cross"], required=True, help="Margin type")

    args = parser.parse_args()

    if args.command == "fetch-indicator":
        sys.exit(cmd_fetch_indicator(args))
    elif args.command == "fetch-factors":
        sys.exit(cmd_fetch_factors(args))
    elif args.command == "risk-xray":
        sys.exit(cmd_risk_xray(args))
    elif args.command == "gap-safe-returns":
        sys.exit(cmd_gap_safe_returns(args))
    elif args.command == "benchmark-returns":
        sys.exit(cmd_benchmark_returns(args))
    elif args.command == "liquidation-price":
        sys.exit(cmd_liquidation_price(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
