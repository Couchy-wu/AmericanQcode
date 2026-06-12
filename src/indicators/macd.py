"""MACD (Moving Average Convergence Divergence) indicator with divergence detection."""

import pandas as pd
import talib

from src.indicators.base import register_indicator, golden_cross, death_cross, detect_divergence


@register_indicator("macd")
def compute_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    column: str = "Close",
    detect_divergence_signal: bool = True,
) -> pd.DataFrame:
    """Compute MACD line, signal line, histogram, crosses, and divergence.

    Args:
        df: DataFrame with OHLCV data.
        fast: Fast EMA period.
        slow: Slow EMA period.
        signal: Signal line EMA period.
        column: Price column.
        detect_divergence_signal: Whether to detect MACD divergence.

    Returns:
        DataFrame with columns: MACD, MACD_Signal, MACD_Histogram,
        MACD_GoldenCross, MACD_DeathCross, MACD_BullishDiv, MACD_BearishDiv.
    """
    result = df.copy()
    close = df[column].values

    macd_line, macd_signal, macd_hist = talib.MACD(
        close, fastperiod=fast, slowperiod=slow, signalperiod=signal
    )

    result["MACD"] = macd_line
    result["MACD_Signal"] = macd_signal
    result["MACD_Histogram"] = macd_hist

    # Cross detection
    macd_series = result["MACD"]
    signal_series = result["MACD_Signal"]
    result["MACD_GoldenCross"] = golden_cross(macd_series, signal_series)
    result["MACD_DeathCross"] = death_cross(macd_series, signal_series)

    # Divergence detection
    if detect_divergence_signal:
        bullish_div, bearish_div = detect_divergence(df[column], macd_series, order=5)
        result["MACD_BullishDiv"] = bullish_div.fillna(False)
        result["MACD_BearishDiv"] = bearish_div.fillna(False)

    return result
