"""VWAP (Volume-Weighted Average Price) indicator.

VWAP is computed as cumulative (Price * Volume) / cumulative Volume.
For intraday use, VWAP resets at the start of each trading day.
For daily data, a rolling window VWAP is provided as an alternative.
"""

import pandas as pd
import numpy as np

from src.indicators.base import register_indicator


@register_indicator("vwap")
def compute_vwap(
    df: pd.DataFrame,
    rolling_window: int | None = None,
    reset_daily: bool = True,
) -> pd.DataFrame:
    """Compute Volume-Weighted Average Price.

    Args:
        df: DataFrame with High, Low, Close, Volume columns.
            Index should be datetime.
        rolling_window: If provided, use a rolling window instead of cumulative.
                        Useful for daily data where intraday reset isn't applicable.
        reset_daily: If True and index has intraday timestamps, reset cumulative
                     calculation at the start of each day. Ignored if rolling_window is set.

    Returns:
        DataFrame with VWAP column.
    """
    result = df.copy()

    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    volume = df["Volume"].values

    # Typical Price
    typical_price = (high + low + close) / 3.0

    if rolling_window:
        # Rolling VWAP (for daily or fixed-window use)
        pv = typical_price * volume
        result["VWAP"] = (
            pd.Series(pv, index=df.index)
            .rolling(window=rolling_window)
            .sum()
            / pd.Series(volume, index=df.index).rolling(window=rolling_window).sum()
        )
    elif reset_daily and hasattr(df.index, "date"):
        # Cumulative VWAP with daily reset (intraday)
        vwap_values = np.zeros(len(df), dtype=np.float64)
        cum_pv = 0.0
        cum_vol = 0.0
        last_date = None

        for i in range(len(df)):
            current_date = df.index[i].date() if hasattr(df.index[i], "date") else None

            if last_date is not None and current_date != last_date:
                cum_pv = 0.0
                cum_vol = 0.0
            last_date = current_date

            cum_pv += typical_price[i] * volume[i]
            cum_vol += volume[i]
            vwap_values[i] = cum_pv / cum_vol if cum_vol > 0 else typical_price[i]

        result["VWAP"] = vwap_values
    else:
        # Simple cumulative VWAP
        cum_pv = np.cumsum(typical_price * volume)
        cum_vol = np.cumsum(volume)
        result["VWAP"] = np.where(cum_vol > 0, cum_pv / cum_vol, typical_price)

    # Price relative to VWAP
    result["VWAP_Above"] = df["Close"].values > result["VWAP"].values
    result["VWAP_Below"] = df["Close"].values < result["VWAP"].values

    return result
