"""Alpha Zoo factor subset — lifted from agent/src/factors/zoo/.

Each factor is a pure function: ``factor(df: pd.DataFrame, **params) -> pd.Series``.
The DataFrame must have at least a ``close`` column (most factors), and some
also require ``high``, ``low``, ``volume``.

Lifted from Microsoft Qlib (Apache-2.0) and alpha101/academic literature.
See individual factor modules for attribution.
"""

from croesus_toolset.factors.momentum import momentum_12_1, roc
from croesus_toolset.factors.risk import volatility_30
from croesus_toolset.factors.reversal import mean_reversion_20
from croesus_toolset.factors.alpha import (
    alpha_momentum_rank,
    alpha_price_volume_divergence,
)
from croesus_toolset.factors.technical import (
    rsi_14,
    macd,
    bollinger_width,
    obv_slope,
    vwap_deviation,
)

from typing import Callable

FACTOR_REGISTRY: dict[str, Callable] = {
    "rsi_14": rsi_14,
    "macd": macd,
    "bollinger_width": bollinger_width,
    "momentum_12_1": momentum_12_1,
    "roc": roc,
    "volatility_30": volatility_30,
    "mean_reversion_20": mean_reversion_20,
    "alpha_momentum_rank": alpha_momentum_rank,
    "alpha_price_volume_divergence": alpha_price_volume_divergence,
    "obv_slope": obv_slope,
    "vwap_deviation": vwap_deviation,
}


def get_factor(name: str) -> callable:
    """Look up a factor by name.

    Raises ``KeyError`` if not found.
    """
    if name not in FACTOR_REGISTRY:
        raise KeyError(
            f"Unknown factor: {name!r}. Available: {sorted(FACTOR_REGISTRY)}"
        )
    return FACTOR_REGISTRY[name]
