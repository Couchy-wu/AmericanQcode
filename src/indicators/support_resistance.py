"""Support and Resistance level detection.

Uses local extrema clustering to identify key horizontal price levels
that have been touched multiple times.
"""

import numpy as np
import pandas as pd
from collections import Counter

from src.indicators.base import register_indicator, find_local_extrema


def _cluster_levels(levels: list[float], tolerance_pct: float = 0.02) -> dict[float, int]:
    """Cluster nearby price levels together.

    Args:
        levels: List of price levels.
        tolerance_pct: Percentage tolerance for merging nearby levels.

    Returns:
        Dict mapping representative price to touch count.
    """
    if not levels:
        return {}

    sorted_levels = sorted(levels)
    clusters: list[list[float]] = []

    for level in sorted_levels:
        found = False
        for cluster in clusters:
            avg = sum(cluster) / len(cluster)
            if abs(level - avg) / avg <= tolerance_pct:
                cluster.append(level)
                found = True
                break
        if not found:
            clusters.append([level])

    return {sum(c) / len(c): len(c) for c in clusters}


@register_indicator("support_resistance")
def compute_support_resistance(
    df: pd.DataFrame,
    window: int = 20,
    min_touches: int = 2,
    tolerance_pct: float = 0.02,
) -> dict:
    """Detect support and resistance levels from price action.

    Uses swing highs (resistance) and swing lows (support) identified
    via local extrema, then clusters nearby levels.

    Args:
        df: DataFrame with High, Low, Close columns.
        window: Number of bars for extrema detection.
        min_touches: Minimum touches for a level to be considered valid.
        tolerance_pct: Price tolerance for level clustering.

    Returns:
        Dict with 'support' and 'resistance' keys, each a list of
        {'level': float, 'touches': int} sorted by touches descending.
    """
    # Use High for resistance, Low for support
    high = df["High"]
    low = df["Low"]

    _, resistance_idx = find_local_extrema(high, order=window)
    support_idx, _ = find_local_extrema(low, order=window)

    resistance_levels = [high.iloc[i] for i in resistance_idx if i < len(high)]
    support_levels = [low.iloc[i] for i in support_idx if i < len(low)]

    # Cluster nearby levels
    resistance_clusters = _cluster_levels(resistance_levels, tolerance_pct)
    support_clusters = _cluster_levels(support_levels, tolerance_pct)

    # Filter by min touches and build result
    resistance = sorted(
        [{"level": round(level, 2), "touches": touches}
         for level, touches in resistance_clusters.items()
         if touches >= min_touches],
        key=lambda x: x["level"], reverse=True,
    )

    support = sorted(
        [{"level": round(level, 2), "touches": touches}
         for level, touches in support_clusters.items()
         if touches >= min_touches],
        key=lambda x: x["level"], reverse=True,
    )

    return {
        "support": support,
        "resistance": resistance,
    }


def add_sr_to_dataframe(
    df: pd.DataFrame,
    sr_data: dict,
) -> pd.DataFrame:
    """Add support/resistance proximity columns to a DataFrame.

    Args:
        df: OHLCV DataFrame.
        sr_data: Output from compute_support_resistance.

    Returns:
        DataFrame with 'Near_Support' and 'Near_Resistance' boolean columns.
    """
    result = df.copy()
    close = df["Close"].values

    support_levels = [s["level"] for s in sr_data.get("support", [])]
    resistance_levels = [r["level"] for r in sr_data.get("resistance", [])]

    if support_levels:
        # For each bar, find distance to nearest support
        nearest_support = np.full(len(close), np.inf)
        for level in support_levels:
            dist_pct = np.abs(close - level) / close
            nearest_support = np.minimum(nearest_support, dist_pct)
        result["Near_Support"] = nearest_support < 0.01  # within 1%
    else:
        result["Near_Support"] = False

    if resistance_levels:
        nearest_resistance = np.full(len(close), np.inf)
        for level in resistance_levels:
            dist_pct = np.abs(close - level) / close
            nearest_resistance = np.minimum(nearest_resistance, dist_pct)
        result["Near_Resistance"] = nearest_resistance < 0.01
    else:
        result["Near_Resistance"] = False

    return result
