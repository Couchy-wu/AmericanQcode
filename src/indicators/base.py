"""Indicator registry and shared utilities (divergence detection, cross detection)."""

from typing import Callable

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

# ─── Indicator Registry ──────────────────────────────────────────────────────

_INDICATOR_REGISTRY: dict[str, Callable] = {}


def register_indicator(name: str):
    """Decorator to register an indicator function in the global registry.

    Usage:
        @register_indicator("macd")
        def compute_macd(df: pd.DataFrame, ...) -> pd.DataFrame:
            ...
    """
    def decorator(fn: Callable):
        _INDICATOR_REGISTRY[name] = fn
        return fn
    return decorator


def get_indicator(name: str) -> Callable | None:
    """Get a registered indicator by name."""
    return _INDICATOR_REGISTRY.get(name)


def list_indicators() -> list[str]:
    """List all registered indicator names."""
    return sorted(_INDICATOR_REGISTRY.keys())


# ─── Cross Detection ─────────────────────────────────────────────────────────


def detect_cross(
    series_a: pd.Series,
    series_b: pd.Series,
    above: bool = True,
) -> pd.Series:
    """Detect where two series cross.

    Args:
        series_a: First series (e.g., short MA).
        series_b: Second series (e.g., long MA).
        above: If True, detects when series_a crosses ABOVE series_b (golden cross).
               If False, detects when series_a crosses BELOW series_b (death cross).

    Returns:
        Boolean Series, True at cross points.
    """
    if above:
        # A was below B in previous bar, and A is now above B
        cross = (series_a > series_b) & (series_a.shift(1) <= series_b.shift(1))
    else:
        # A was above B in previous bar, and A is now below B
        cross = (series_a < series_b) & (series_a.shift(1) >= series_b.shift(1))
    return cross.fillna(False)


def golden_cross(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    """Detect golden cross: A crosses above B."""
    return detect_cross(series_a, series_b, above=True)


def death_cross(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    """Detect death cross: A crosses below B."""
    return detect_cross(series_a, series_b, above=False)


# ─── Divergence Detection ────────────────────────────────────────────────────


def find_local_extrema(series: pd.Series, order: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Find local minima and maxima indices in a series.

    Args:
        series: The data series.
        order: Number of points on each side to consider for extrema.

    Returns:
        (minima_indices, maxima_indices) as numpy arrays.
    """
    # Drop NaN values for extrema detection
    clean = series.dropna()
    if len(clean) < order * 2 + 1:
        return np.array([]), np.array([])

    minima_idx = argrelextrema(clean.values, np.less, order=order)[0]
    maxima_idx = argrelextrema(clean.values, np.greater, order=order)[0]

    # Map back to original index positions
    minima_mapped = np.array([clean.index.get_loc(clean.index[i]) for i in minima_idx])
    maxima_mapped = np.array([clean.index.get_loc(clean.index[i]) for i in maxima_idx])

    return minima_mapped, maxima_mapped


def detect_divergence(
    price: pd.Series,
    indicator: pd.Series,
    order: int = 5,
    lookback: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Detect bullish and bearish divergence between price and an indicator.

    Bullish divergence: Price makes a lower low but indicator makes a higher low.
    Bearish divergence: Price makes a higher high but indicator makes a lower high.

    Args:
        price: Price series (typically Close).
        indicator: Indicator series (e.g., RSI, MACD).
        order: Extrema detection sensitivity.
        lookback: How many recent extrema to compare.

    Returns:
        (bullish_mask, bearish_mask) — boolean Series aligned with the original index.
    """
    n = len(price)
    bullish = pd.Series(False, index=price.index)
    bearish = pd.Series(False, index=price.index)

    if n < order * 2 + 3:
        return bullish, bearish

    # Find extrema in both series
    p_min_idx, p_max_idx = find_local_extrema(price, order)
    i_min_idx, i_max_idx = find_local_extrema(indicator, order)

    # Bullish divergence: compare last `lookback` price minima with indicator minima
    for i in range(1, min(lookback + 1, len(p_min_idx))):
        p_prev_idx = p_min_idx[-i - 1] if len(p_min_idx) > i else None
        p_curr_idx = p_min_idx[-1] if len(p_min_idx) > 0 else None
        i_prev_idx = i_min_idx[-i - 1] if len(i_min_idx) > i else None
        i_curr_idx = i_min_idx[-1] if len(i_min_idx) > 0 else None

        if p_prev_idx is None or p_curr_idx is None or i_prev_idx is None or i_curr_idx is None:
            continue

        if price.iloc[p_curr_idx] < price.iloc[p_prev_idx] and \
           indicator.iloc[i_curr_idx] > indicator.iloc[i_prev_idx]:
            bullish.iloc[p_curr_idx] = True

    # Bearish divergence: compare last `lookback` price maxima with indicator maxima
    for i in range(1, min(lookback + 1, len(p_max_idx))):
        p_prev_idx = p_max_idx[-i - 1] if len(p_max_idx) > i else None
        p_curr_idx = p_max_idx[-1] if len(p_max_idx) > 0 else None
        i_prev_idx = i_max_idx[-i - 1] if len(i_max_idx) > i else None
        i_curr_idx = i_max_idx[-1] if len(i_max_idx) > 0 else None

        if p_prev_idx is None or p_curr_idx is None or i_prev_idx is None or i_curr_idx is None:
            continue

        if price.iloc[p_curr_idx] > price.iloc[p_prev_idx] and \
           indicator.iloc[p_curr_idx] < indicator.iloc[i_prev_idx]:
            bearish.iloc[p_curr_idx] = True

    return bullish, bearish


# ─── Utility: Ensure required columns ─────────────────────────────────────────


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise ValueError if required columns are missing."""
    available = set(col.lower() for col in df.columns)
    for col in columns:
        if col.lower() not in available:
            raise ValueError(f"DataFrame missing required column: '{col}'")
