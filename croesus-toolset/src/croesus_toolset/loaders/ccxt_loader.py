"""CCXT loader — wraps ``ccxt`` for exchange-routed pairs.

No API key required for public spot candles.  Uses ``ccxt.<exchange>({})``
with rate limiting enabled.

Usage::

    loader = CcxtLoader()
    df = loader.fetch_ohlcv(
        "BTC/USDT",
        interval="1h",
        window="90d",
        exchange="binance",
    )
"""

from __future__ import annotations

import pandas as pd

from croesus_toolset.loaders.base import BaseLoader

# Map human-friendly intervals to ccxt timeframe codes
_INTERVAL_MAP: dict[str, str] = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
    "1w": "1w",
}

# Rough mapping from window strings to candle counts
_WINDOW_LIMIT: dict[str, int] = {
    "1d": 1,
    "7d": 7,
    "30d": 30,
    "60d": 60,
    "90d": 90,
    "180d": 180,
    "365d": 365,
}


def _window_to_limit(window: str) -> int:
    """Convert a window string like ``'90d'`` to a candle count."""
    w = window.strip().lower()
    if w in _WINDOW_LIMIT:
        return _WINDOW_LIMIT[w]
    # Try numeric prefix
    num = "".join(c for c in w if c.isdigit())
    if num:
        return int(num)
    return 200  # sensible default


class CcxtLoader(BaseLoader):
    """Fetch OHLCV from an exchange via ccxt."""

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str = "1d",
        window: str = "90d",
        **kwargs,
    ) -> pd.DataFrame:
        try:
            import ccxt
        except ImportError:
            raise ImportError(
                "ccxt is required for the ccxt loader. "
                "Install with: pip install 'croesus-toolset[fetch]'"
            )

        exchange_id = kwargs.get("exchange", "binance")
        exchange_class = getattr(ccxt, exchange_id, None)
        if exchange_class is None:
            raise ValueError(f"Unknown exchange: {exchange_id!r}")

        exchange = exchange_class({"enableRateLimit": True})
        timeframe = _INTERVAL_MAP.get(interval, interval)
        limit = _window_to_limit(window)

        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except Exception as exc:
            raise ValueError(
                f"Failed to fetch {symbol!r} from {exchange_id}: {exc}"
            ) from exc

        if not ohlcv:
            raise ValueError(f"No data returned for {symbol!r} from {exchange_id}")

        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp")
        return df
