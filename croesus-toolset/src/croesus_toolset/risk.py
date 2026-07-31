"""Portfolio risk x-ray — concentration, volatility, drawdown, tail risk.

Lifted from Vibe-Trading agent/backtest/risk_xray.py.
Pure functions of prices and weights — no I/O, no network, no loader imports.
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

MIN_HISTORY_DAYS = 30
PERIODS_PER_YEAR = 252
VAR_LEVELS = (0.95, 0.99)


def _finite(value: float | None) -> float | None:
    """Return ``value`` when finite, else ``None`` (strict-JSON safe)."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _validate_weights(
    closes: pd.DataFrame, weights: Mapping[str, float]
) -> tuple[dict[str, float], list[str]]:
    """Normalize weights to sum 1; reject unknown symbols and bad values."""
    warnings: list[str] = []
    if not weights:
        raise ValueError("weights must name at least one symbol")

    unknown = [sym for sym in weights if sym not in closes.columns]
    if unknown:
        raise ValueError(f"weights reference symbols with no price data: {sorted(unknown)}")

    cleaned: dict[str, float] = {}
    for sym, raw in weights.items():
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"weight for {sym!r} is not a number: {raw!r}") from exc
        if not math.isfinite(value):
            raise ValueError(f"weight for {sym!r} is not finite: {raw!r}")
        if value < 0:
            raise ValueError(
                f"weight for {sym!r} is negative ({value}); the risk x-ray is long-only for now"
            )
        cleaned[sym] = value

    total = sum(cleaned.values())
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    if abs(total - 1.0) > 1e-6:
        warnings.append(f"weights summed to {total:.6f}; renormalized to 1.0")
        cleaned = {sym: value / total for sym, value in cleaned.items()}
    return cleaned, warnings


def compute_risk_xray(
    closes: pd.DataFrame,
    weights: Mapping[str, float],
    *,
    periods_per_year: int = PERIODS_PER_YEAR,
    var_levels: Sequence[float] = VAR_LEVELS,
    min_history: int = MIN_HISTORY_DAYS,
) -> dict[str, Any]:
    """Compute the risk x-ray for a weighted basket.

    Args:
        closes: Close-price panel, one column per symbol, sorted by date.
        weights: Symbol -> weight. Renormalized to 1.0 with a warning when the
            sum differs; must be long-only and reference existing columns.
        periods_per_year: Annualization factor for the bar interval.
        var_levels: Tail levels for historical VaR / expected shortfall.
        min_history: Minimum valid bars a symbol must have to be included.

    Returns:
        A strict-JSON-safe dict with concentration, volatility, drawdown,
        tail risk, diversification, and correlation sections.
    """
    if closes is None or closes.empty:
        raise ValueError("price panel is empty")
    frame = closes.dropna(axis=1, how="all")
    if frame.empty:
        raise ValueError("price panel has no non-NaN closes")

    weights, warnings = _validate_weights(frame, weights)

    # History filter
    kept: list[str] = []
    skipped: list[dict[str, str]] = []
    for sym in weights:
        valid = int(frame[sym].count())
        if valid < min_history:
            skipped.append({"symbol": sym, "reason": f"only {valid} valid bars (min {min_history})"})
        else:
            kept.append(sym)
    if not kept:
        raise ValueError(f"no symbol has at least {min_history} valid bars")
    if skipped:
        kept_weights = {sym: weights[sym] for sym in kept}
        total = sum(kept_weights.values())
        if total <= 0:
            raise ValueError("surviving symbols have zero total weight")
        weights = {sym: value / total for sym, value in kept_weights.items()}
        warnings.append("weights renormalized over symbols that survived the history filter")

    aligned = frame[kept].dropna(axis=0, how="any")
    if len(aligned) < 2:
        raise ValueError(
            "fewer than 2 shared trading days after aligning calendars across symbols"
        )

    returns = aligned.pct_change(fill_method=None).dropna(how="any")
    if returns.empty:
        raise ValueError("no overlapping return observations across symbols")

    w = np.array([weights[sym] for sym in kept], dtype=float)
    port = returns.to_numpy(dtype=float) @ w
    port_returns = pd.Series(port, index=returns.index)

    result: dict[str, Any] = {
        "inputs": {
            "symbols": kept,
            "weights": {sym: round(weights[sym], 6) for sym in kept},
            "bars": len(aligned),
        },
        "warnings": warnings,
        "skipped": skipped,
    }

    # ── Concentration ──────────────────────────────────────────────────────
    sq = np.array([weights[sym] ** 2 for sym in kept])
    hhi = float(sq.sum())
    effective_n = 1.0 / hhi if hhi > 0 else 0.0
    result["concentration"] = {
        "hhi": round(hhi, 6),
        "effective_n": round(effective_n, 4),
    }

    # ── Volatility ────────────────────────────────────────────────────────
    ann_vol = float(port_returns.std() * np.sqrt(periods_per_year))
    result["volatility"] = {
        "annualized_vol": round(_finite(ann_vol) or 0.0, 6),
        "daily_vol": round(float(port_returns.std()), 6),
    }

    # ── Drawdown ──────────────────────────────────────────────────────────
    equity = (1 + port_returns).cumprod()
    peak = equity.cummax()
    dd = (equity - peak) / peak.replace(0, np.nan)
    max_dd = float(dd.min()) if not dd.empty else 0.0
    result["drawdown"] = {
        "max_drawdown": round(_finite(max_dd) or 0.0, 6),
    }

    # ── Tail risk (VaR / ES) ─────────────────────────────────────────────
    tail: dict[str, Any] = {}
    for level in var_levels:
        alpha = 1 - level
        var_val = float(np.percentile(port_returns, alpha * 100))
        es_mask = port_returns <= var_val
        es_val = float(port_returns[es_mask].mean()) if es_mask.any() else var_val
        tail[f"var_{int(level * 100)}"] = round(_finite(var_val) or 0.0, 6)
        tail[f"es_{int(level * 100)}"] = round(_finite(es_val) or 0.0, 6)
    result["tail_risk"] = tail

    # ── Diversification ──────────────────────────────────────────────────
    corr = returns.corr()
    avg_corr = 0.0
    n = len(kept)
    if n > 1:
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                c = corr.loc[kept[i], kept[j]]
                if math.isfinite(c):
                    pairs.append(c)
        avg_corr = float(np.mean(pairs)) if pairs else 0.0
    result["diversification"] = {
        "avg_pairwise_corr": round(avg_corr, 6),
    }

    # ── Correlation matrix ────────────────────────────────────────────────
    result["correlation"] = {
        sym1: {sym2: round(float(corr.loc[sym1, sym2]), 4) for sym2 in kept}
        for sym1 in kept
    }

    return result


def render_risk_xray_markdown(report: dict[str, Any]) -> str:
    """Render a risk x-ray report as Markdown."""
    lines = ["# Portfolio Risk X-Ray", ""]

    inp = report.get("inputs", {})
    lines.append(f"**Symbols:** {', '.join(inp.get('symbols', []))}")
    lines.append(f"**Bars:** {inp.get('bars', 'N/A')}")
    lines.append("")

    conc = report.get("concentration", {})
    lines.append("## Concentration")
    lines.append(f"- HHI: {conc.get('hhi', 'N/A')}")
    lines.append(f"- Effective N: {conc.get('effective_n', 'N/A')}")
    lines.append("")

    vol = report.get("volatility", {})
    lines.append("## Volatility")
    lines.append(f"- Annualized vol: {vol.get('annualized_vol', 'N/A')}")
    lines.append(f"- Daily vol: {vol.get('daily_vol', 'N/A')}")
    lines.append("")

    dd = report.get("drawdown", {})
    lines.append("## Drawdown")
    lines.append(f"- Max drawdown: {dd.get('max_drawdown', 'N/A')}")
    lines.append("")

    tail = report.get("tail_risk", {})
    if tail:
        lines.append("## Tail Risk")
        for k, v in tail.items():
            lines.append(f"- {k}: {v}")
        lines.append("")

    div = report.get("diversification", {})
    lines.append("## Diversification")
    lines.append(f"- Avg pairwise corr: {div.get('avg_pairwise_corr', 'N/A')}")
    lines.append("")

    for w in report.get("warnings", []):
        lines.append(f"> ⚠️ {w}")

    return "\n".join(lines)
