"""OBV (On-Balance Volume) indicator.

OBV accumulates volume on up days and subtracts volume on down days.
Divergence between OBV and price can signal trend reversals.
"""

import pandas as pd
import talib

from src.indicators.base import register_indicator, detect_divergence


@register_indicator("obv")
def compute_obv(
    df: pd.DataFrame,
    sma_period: int = 20,
    detect_divergence_signal: bool = True,
) -> pd.DataFrame:
    """Compute On-Balance Volume with optional signal line and divergence.

    Args:
        df: DataFrame with Close and Volume columns.
        sma_period: SMA period for OBV signal line.
        detect_divergence_signal: Whether to detect OBV divergence.

    Returns:
        DataFrame with OBV and OBV_SMA columns.
    """
    result = df.copy()

    close = df["Close"].values
    volume = df["Volume"].values

    obv = talib.OBV(close, volume)
    result["OBV"] = obv

    # OBV moving average as signal line
    result["OBV_SMA"] = talib.SMA(obv, timeperiod=sma_period)

    # OBV direction
    obv_series = pd.Series(obv, index=df.index)
    result["OBV_Rising"] = obv_series > obv_series.shift(1)
    result["OBV_Falling"] = obv_series < obv_series.shift(1)

    # Divergence
    if detect_divergence_signal and len(df) > sma_period + 10:
        bullish_div, bearish_div = detect_divergence(close, pd.Series(obv, index=df.index), order=5)
        result["OBV_BullishDiv"] = bullish_div.fillna(False)
        result["OBV_BearishDiv"] = bearish_div.fillna(False)

    return result
