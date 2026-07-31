"""CLI entrypoint for croesus-toolset.

Commands:
    fetch-indicator  — fetch OHLCV data and compute technical indicators
    risk-xray        — compute portfolio risk x-ray from a holdings JSON file

Usage:
    croesus fetch-indicator --asset BTCUSDT --indicator rsi_14 --interval 1h
    croesus risk-xray --portfolio ./holdings.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _fetch_ohlcv_yfinance(symbol: str, interval: str, days: int = 30):
    """Fetch OHLCV via yfinance. Returns a pd.DataFrame or None."""
    try:
        import yfinance as yf
    except ImportError:
        return None

    # Map our interval names to yfinance
    yf_interval = {
        "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1h", "1H": "1h", "1d": "1d", "1wk": "1wk", "1mo": "1mo",
    }.get(interval, "1d")

    ticker = yf.Ticker(symbol)
    df = ticker.history(period=f"{days}d", interval=yf_interval)
    if df.empty:
        return None
    return df


def _fetch_ohlcv_ccxt(exchange_id: str, symbol: str, timeframe: str, limit: int = 200):
    """Fetch OHLCV via ccxt. Returns a list of [ts, o, h, l, c, v] or None."""
    try:
        import ccxt
    except ImportError:
        return None

    exchange_class = getattr(ccxt, exchange_id, None)
    if exchange_class is None:
        return None
    exchange = exchange_class({"enableRateLimit": True})
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        return ohlcv
    except Exception:
        return None


def _detect_fetch_method(symbol: str, interval: str):
    """Return (method, args) for the best fetch method.

    Strategy: try yfinance first for all symbols (broadest compatibility).
    ccxt is available as a fallback for crypto when yfinance fails.
    """
    # All symbols go through yfinance first — it handles stocks, crypto, forex
    return "yfinance", (symbol, interval)


def cmd_fetch_indicator(args):
    """Fetch OHLCV and compute technical indicators."""
    from croesus_toolset.indicators import (
        compute_bollinger, compute_ema, compute_macd, compute_rsi, compute_sma,
    )

    symbol = args.asset
    indicator = args.indicator
    interval = args.interval

    # Fetch data
    method, fetch_args = _detect_fetch_method(symbol, interval)

    if method == "ccxt":
        raw = _fetch_ohlcv_ccxt(*fetch_args)
        if raw is None:
            print(json.dumps({"ok": False, "error": f"Failed to fetch data for {symbol} via ccxt"}))
            return 1
        import pandas as pd
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        close = df["close"]
    else:
        df = _fetch_ohlcv_yfinance(*fetch_args)
        if df is None or df.empty:
            print(json.dumps({"ok": False, "error": f"No data returned for {symbol}"}))
            return 1
        close = df["Close"] if "Close" in df.columns else df.get("close")
        if close is None:
            print(json.dumps({"ok": False, "error": "No close price column in data"}))
            return 1

    # Compute requested indicator
    indicator_funcs = {
        "rsi_14": lambda c: compute_rsi(c, period=14),
        "rsi": lambda c: compute_rsi(c, period=14),
        "macd": compute_macd,
        "bollinger": compute_bollinger,
        "sma_20": lambda c: compute_sma(c, period=20),
        "sma_50": lambda c: compute_sma(c, period=50),
        "sma_200": lambda c: compute_sma(c, period=200),
        "ema_20": lambda c: compute_ema(c, period=20),
    }

    func = indicator_funcs.get(indicator)
    if func is None:
        print(json.dumps({"ok": False, "error": f"Unknown indicator: {indicator}. Available: {list(indicator_funcs.keys())}"}))
        return 1

    result = func(close)
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

    # Expect {"weights": {"SYM": 0.5, ...}, "symbols": ["SYM", ...]}
    weights = portfolio.get("weights")
    symbols = portfolio.get("symbols", list(weights.keys()) if weights else [])

    if not weights:
        print(json.dumps({"ok": False, "error": "Portfolio must have 'weights' key"}))
        return 1

    # Fetch close prices for each symbol
    import pandas as pd
    closes_dict = {}
    for sym in symbols:
        method, fetch_args = _detect_fetch_method(sym, args.interval)
        if method == "ccxt":
            raw = _fetch_ohlcv_ccxt(*fetch_args)
            if raw:
                df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
                closes_dict[sym] = df["close"].reset_index(drop=True)
        else:
            df = _fetch_ohlcv_yfinance(*fetch_args)
            if df is not None and not df.empty:
                col = "Close" if "Close" in df.columns else "close"
                if col:
                    closes_dict[sym] = df[col].reset_index(drop=True)

    if not closes_dict:
        print(json.dumps({"ok": False, "error": "No price data fetched for any symbol"}))
        return 1

    closes = pd.DataFrame(closes_dict)

    try:
        report = compute_risk_xray(closes, weights)
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1

    if args.format == "markdown":
        print(render_risk_xray_markdown(report))
    else:
        print(json.dumps(report, ensure_ascii=False, default=str))
    return 0




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


def _fetch_close_prices(symbol: str, interval: str, days: int = 30):
    """Fetch close prices for a symbol. Returns pd.Series or None."""
    try:
        df = _fetch_ohlcv_yfinance(symbol, interval, days)
        if df is None or df.empty:
            return None
        if "Close" in df.columns:
            return df["Close"].reset_index(drop=True)
        elif "close" in df.columns:
            return df["close"].reset_index(drop=True)
    except (ImportError, ValueError):
        pass
    return None


def cmd_gap_safe_returns(args):
    """Compute gap-safe bar returns for an asset."""
    from croesus_toolset.backtest import gap_safe_bar_returns
    close = _fetch_close_prices(args.asset, args.interval, days=_parse_window(args.window))
    if close is None or len(close) < 2:
        print(json.dumps({"ok": False, "error": f"Insufficient data for {args.asset}"}))
        return 1
    result = gap_safe_bar_returns(close, halted_threshold=args.threshold)
    halted_mask = result.attrs.get("halted_mask", None)
    output = {
        "ok": True, "symbol": args.asset, "interval": args.interval,
        "window": args.window, "halted_threshold": args.threshold,
        "bars": len(result), "returns": result.tolist(),
        "halted_bars": int(halted_mask.sum()) if halted_mask is not None else 0,
        "mean_return": float(result.mean()), "std_return": float(result.std()),
    }
    print(json.dumps(output, ensure_ascii=False, default=str))
    return 0


def cmd_benchmark_returns(args):
    """Compute sign-safe benchmark returns for an asset."""
    from croesus_toolset.backtest import sign_safe_benchmark
    close = _fetch_close_prices(args.asset, args.interval, days=_parse_window(args.window))
    if close is None or len(close) < 2:
        print(json.dumps({"ok": False, "error": f"Insufficient data for {args.asset}"}))
        return 1
    result = sign_safe_benchmark(close)
    total_return = float((1 + result).prod() - 1)
    output = {
        "ok": True, "symbol": args.asset, "interval": args.interval,
        "window": args.window, "bars": len(result), "total_return": total_return,
        "mean_return": float(result.mean()), "std_return": float(result.std()),
    }
    print(json.dumps(output, ensure_ascii=False, default=str))
    return 0


def cmd_liquidation_price(args):
    """Compute USD-M perpetual liquidation price."""
    from croesus_toolset.backtest import usdm_liquidation_price
    try:
        liq = usdm_liquidation_price(args.entry, args.leverage, args.side, args.margin)
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1
    output = {
        "ok": True, "entry_price": args.entry, "leverage": args.leverage,
        "side": args.side, "margin_type": args.margin,
        "liquidation_price": liq, "distance_pct": abs(liq - args.entry) / args.entry * 100,
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
    fi.add_argument("--indicator", required=True, help="Indicator to compute (rsi_14, macd, bollinger, sma_20, etc.)")
    fi.add_argument("--interval", default="1d", help="Bar interval (1h, 1d, etc.)")

    # risk-xray
    rx = sub.add_parser("risk-xray", help="Compute portfolio risk x-ray")
    rx.add_argument("--portfolio", required=True, help="Path to holdings JSON file")
    rx.add_argument("--interval", default="1d", help="Bar interval for price data")
    rx.add_argument("--format", choices=["json", "markdown"], default="json", help="Output format")

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
