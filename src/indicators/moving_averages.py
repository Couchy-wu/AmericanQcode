"""Simple Moving Average (SMA) and Exponential Moving Average (EMA) indicators."""

import pandas as pd
import talib
import numpy as np

from src.indicators.base import register_indicator, golden_cross, death_cross


@register_indicator("sma")
def compute_sma(df: pd.DataFrame, period: int = 20, column: str = "Close") -> pd.DataFrame:
    """Compute Simple Moving Average."""
    result = df.copy()
    result[f"SMA_{period}"] = talib.SMA(df[column].values, timeperiod=period)
    return result


@register_indicator("ema")
def compute_ema(df: pd.DataFrame, period: int = 20, column: str = "Close") -> pd.DataFrame:
    """Compute Exponential Moving Average."""
    result = df.copy()
    result[f"EMA_{period}"] = talib.EMA(df[column].values, timeperiod=period)
    return result


@register_indicator("ma_cross")
def compute_ma_cross(
    df: pd.DataFrame,
    fast: int = 20,
    slow: int = 50,
    column: str = "Close",
    ma_type: str = "ema",
) -> pd.DataFrame:
    """Compute fast and slow MAs and detect golden/death crosses.

    Args:
        df: DataFrame with OHLCV data.
        fast: Fast MA period.
        slow: Slow MA period.
        column: Price column to use.
        ma_type: 'sma' or 'ema'.

    Returns:
        DataFrame with MA columns and cross signals.
    """
    result = df.copy()
    close = df[column].values

    if ma_type == "sma":
        result[f"MA_{fast}"] = talib.SMA(close, timeperiod=fast)
        result[f"MA_{slow}"] = talib.SMA(close, timeperiod=slow)
    else:
        result[f"MA_{fast}"] = talib.EMA(close, timeperiod=fast)
        result[f"MA_{slow}"] = talib.EMA(close, timeperiod=slow)

    fast_ma = result[f"MA_{fast}"]
    slow_ma = result[f"MA_{slow}"]

    result["MA_GoldenCross"] = golden_cross(fast_ma, slow_ma)
    result["MA_DeathCross"] = death_cross(fast_ma, slow_ma)

    return result
