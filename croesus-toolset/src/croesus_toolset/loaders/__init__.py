"""Data-source loaders for croesus-toolset.

Each loader implements the ``BaseLoader`` interface and returns a standard
OHLCV ``pd.DataFrame`` with lowercase column names:
open, high, low, close, volume.

Usage::

    from croesus_toolset.loaders import get_loader

    loader = get_loader("yfinance")
    df = loader.fetch_ohlcv("BTC-USD", interval="1d", window="90d")
"""

from croesus_toolset.loaders.base import BaseLoader
from croesus_toolset.loaders.ccxt_loader import CcxtLoader
from croesus_toolset.loaders.yfinance_loader import YFinanceLoader

_LOADER_MAP: dict[str, type[BaseLoader]] = {
    "yfinance": YFinanceLoader,
    "ccxt": CcxtLoader,
}


def get_loader(name: str) -> BaseLoader:
    """Return an instance of the named loader.

    Raises ``ValueError`` for unknown loader names.
    """
    cls = _LOADER_MAP.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown loader: {name!r}. Available: {sorted(_LOADER_MAP)}"
        )
    return cls()
