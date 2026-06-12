"""RSI (Relative Strength Index) indicator with divergence detection."""

import pandas as pd
import talib

from src.indicators.base import register_indicator, detect_divergence


@register_indicator("rsi")
def compute_rsi(
    df: pd.DataFrame,
    period: int = 14,
    column: str = "Close",
    oversold: float = 30.0,
    overbought: float = 70.0,
    detect_divergence_signal: bool = True,
) -> pd.DataFrame:
    """Compute RSI, overbought/oversold signals, and divergence.

    Args:
        df: DataFrame with OHLCV data.
        period: RSI lookback period.
        column: Price column.
        oversold: Oversold threshold.
        overbought: Overbought threshold.
        detect_divergence_signal: Whether to detect RSI divergence.

    Returns:
        DataFrame with RSI, overbought/oversold flags, and divergence signals.
    """
    result = df.copy()
    close = df[column].values

    rsi_values = talib.RSI(close, timeperiod=period)
    result["RSI"] = rsi_values

    # Overbought / Oversold regions
    result["RSI_Oversold"] = rsi_values < oversold
    result["RSI_Overbought"] = rsi_values > overbought

    # Cross above oversold (bullish) and cross below overbought (bearish)
    rsi_series = pd.Series(rsi_values, index=df.index)
    result["RSI_ExitOversold"] = (rsi_series > oversold) & (rsi_series.shift(1) <= oversold)
    result["RSI_ExitOverbought"] = (rsi_series < overbought) & (rsi_series.shift(1) >= overbought)

    # Divergence
    if detect_divergence_signal and len(df) > period + 10:
        bullish_div, bearish_div = detect_divergence(df[column], rsi_series, order=5)
        result["RSI_BullishDiv"] = bullish_div.fillna(False)
        result["RSI_BearishDiv"] = bearish_div.fillna(False)

    return result
