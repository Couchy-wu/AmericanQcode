"""ADX (Average Directional Index) indicator.

ADX measures trend strength regardless of direction.
+DI and -DI indicate directional bias.
"""

import pandas as pd
import talib

from src.indicators.base import register_indicator


@register_indicator("adx")
def compute_adx(
    df: pd.DataFrame,
    period: int = 14,
    trend_threshold: float = 25.0,
    strong_trend_threshold: float = 40.0,
) -> pd.DataFrame:
    """Compute ADX, +DI, and -DI.

    Args:
        df: DataFrame with High, Low, Close columns.
        period: ADX lookback period (default 14).
        trend_threshold: ADX above this = trending (default 25).
        strong_trend_threshold: ADX above this = strong trend (default 40).

    Returns:
        DataFrame with ADX, +DI, -DI columns and trend strength flags.
    """
    result = df.copy()

    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values

    adx = talib.ADX(high, low, close, timeperiod=period)
    plus_di = talib.PLUS_DI(high, low, close, timeperiod=period)
    minus_di = talib.MINUS_DI(high, low, close, timeperiod=period)

    result["ADX"] = adx
    result["DI_Plus"] = plus_di
    result["DI_Minus"] = minus_di

    # Trend strength classification
    result["ADX_Trending"] = adx > trend_threshold
    result["ADX_StrongTrend"] = adx > strong_trend_threshold
    result["ADX_Ranging"] = adx <= trend_threshold

    # Directional bias
    result["ADX_Bullish"] = (plus_di > minus_di) & (adx > trend_threshold)
    result["ADX_Bearish"] = (minus_di > plus_di) & (adx > trend_threshold)

    # Crossovers
    plus_series = pd.Series(plus_di, index=df.index)
    minus_series = pd.Series(minus_di, index=df.index)
    result["ADX_BullishCross"] = (plus_series > minus_series) & (plus_series.shift(1) <= minus_series.shift(1))
    result["ADX_BearishCross"] = (minus_series > plus_series) & (minus_series.shift(1) <= plus_series.shift(1))

    return result
