"""YFinance loader — wraps ``yfinance`` for public tickers.

No API key required. Handles stocks, crypto, forex via Yahoo Finance.
"""

from __future__ import annotations

import pandas as pd

from croesus_toolset.loaders.base import BaseLoader

# Map human-friendly intervals to yfinance interval codes
_INTERVAL_MAP: dict[str, str] = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "1H": "1h",
    "1d": "1d",
    "1wk": "1wk",
    "1mo": "1mo",
}


class YFinanceLoader(BaseLoader):
    """Fetch OHLCV from Yahoo Finance via yfinance."""

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str = "1d",
        window: str = "90d",
        **kwargs,
    ) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError(
                "yfinance is required for the yfinance loader. "
                "Install with: pip install 'croesus-toolset[fetch]'"
            )

        yf_interval = _INTERVAL_MAP.get(interval, "1d")
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=window, interval=yf_interval)

        if df.empty:
            raise ValueError(f"No data returned for {symbol!r}")

        # Normalise column names to lowercase
        df.columns = [c.lower() for c in df.columns]
        return df
