"""Abstract base loader interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseLoader(ABC):
    """Thin interface every data-source loader must satisfy."""

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str = "1d",
        window: str = "90d",
        **kwargs,
    ) -> pd.DataFrame:
        """Fetch OHLCV candles and return a DataFrame with columns:

        open, high, low, close, volume

        Parameters
        ----------
        symbol : str
            Ticker or pair (e.g. ``"BTC-USD"`` or ``"BTC/USDT"``).
        interval : str
            Bar interval (``"1h"``, ``"1d"``, etc.).
        window : str
            How far back to fetch (``"90d"``, ``"6mo"``, etc.).
        **kwargs
            Loader-specific options (e.g. ``exchange`` for ccxt).
        """
        ...
