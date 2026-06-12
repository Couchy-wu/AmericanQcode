"""Bollinger Bands indicator with squeeze detection."""

import pandas as pd
import numpy as np
import talib

from src.indicators.base import register_indicator


@register_indicator("bollinger")
def compute_bollinger(
    df: pd.DataFrame,
    period: int = 20,
    nbdev: float = 2.0,
    column: str = "Close",
) -> pd.DataFrame:
    """Compute Bollinger Bands (upper, middle, lower).

    Also computes:
        %B — where price sits within the bands (0 = lower, 1 = upper).
        Bandwidth — (upper - lower) / middle (used for squeeze detection).
        Squeeze — True when bandwidth is at a local minimum.

    Args:
        df: DataFrame with OHLCV data.
        period: Moving average period.
        nbdev: Number of standard deviations for bands.
        column: Price column.

    Returns:
        DataFrame with BB_Upper, BB_Middle, BB_Lower, BB_PctB, BB_Bandwidth, BB_Squeeze.
    """
    result = df.copy()
    close = df[column].values

    upper, middle, lower = talib.BBANDS(close, timeperiod=period, nbdevup=nbdev,
                                         nbdevdn=nbdev, matype=0)

    result["BB_Upper"] = upper
    result["BB_Middle"] = middle
    result["BB_Lower"] = lower

    # %B: where price sits within the bands
    # %B = (Price - Lower) / (Upper - Lower)
    diff = upper - lower
    result["BB_PctB"] = np.where(diff > 0, (close - lower) / diff, 0.5)

    # Bandwidth: (Upper - Lower) / Middle
    result["BB_Bandwidth"] = np.where(middle > 0, diff / middle, 0)

    # Squeeze detection: bandwidth is below its 20-period low percentile
    bw_series = pd.Series(result["BB_Bandwidth"], index=df.index)
    squeeze_threshold = bw_series.rolling(window=period * 5).quantile(0.1)
    result["BB_Squeeze"] = bw_series < squeeze_threshold

    # Price crossing bands
    close_series = pd.Series(close, index=df.index)
    result["BB_CrossUpper"] = close_series > upper
    result["BB_CrossLower"] = close_series < lower

    return result
