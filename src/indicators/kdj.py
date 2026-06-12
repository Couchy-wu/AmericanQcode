"""KDJ (Stochastic Oscillator) indicator.

KDJ is derived from the Stochastic indicator, adding the J line for
earlier signals. Commonly used in Chinese/Asian markets alongside MACD.
"""

import pandas as pd
import talib

from src.indicators.base import register_indicator, golden_cross, death_cross


@register_indicator("kdj")
def compute_kdj(
    df: pd.DataFrame,
    n: int = 9,
    m1: int = 3,
    m2: int = 3,
) -> pd.DataFrame:
    """Compute KDJ indicator.

    KDJ formula:
        RSV = (Close - LowestLow_n) / (HighestHigh_n - LowestLow_n) * 100
        K = EMA(RSV, m1)
        D = EMA(K, m2)
        J = 3 * K - 2 * D

    Args:
        df: DataFrame with OHLCV data (requires High, Low, Close columns).
        n: RSV lookback period (typically 9).
        m1: K line smoothing period (typically 3).
        m2: D line smoothing period (typically 3).

    Returns:
        DataFrame with K, D, J columns, plus golden/death cross signals.
    """
    result = df.copy()

    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values

    # Use TA-Lib's STOCH for K and D
    slowk, slowd = talib.STOCH(high, low, close, fastk_period=n, slowk_period=m1,
                               slowk_matype=1, slowd_period=m2, slowd_matype=1)

    # J = 3 * K - 2 * D
    j = 3 * slowk - 2 * slowd

    result["KDJ_K"] = slowk
    result["KDJ_D"] = slowd
    result["KDJ_J"] = j

    # Cross signals
    k_series = pd.Series(slowk, index=df.index)
    d_series = pd.Series(slowd, index=df.index)
    result["KDJ_GoldenCross"] = golden_cross(k_series, d_series)
    result["KDJ_DeathCross"] = death_cross(k_series, d_series)

    # Overbought / Oversold on J line
    j_series = pd.Series(j, index=df.index)
    result["KDJ_Oversold"] = j_series < 20
    result["KDJ_Overbought"] = j_series > 80

    return result
