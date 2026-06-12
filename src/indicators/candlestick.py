"""Candlestick pattern recognition using TA-Lib.

TA-Lib provides 61 candlestick pattern recognition functions.
Each function returns an integer: positive = bullish, negative = bearish, 0 = no pattern.

Reference: TA-Lib Pattern Recognition functions.
"""

from functools import lru_cache
from typing import Callable

import pandas as pd
import talib

from src.indicators.base import register_indicator

# ─── Pattern Registry ────────────────────────────────────────────────────────

# All available TA-Lib candlestick patterns
_PATTERNS: dict[str, Callable] = {
    "CDL2CROWS": talib.CDL2CROWS,
    "CDL3BLACKCROWS": talib.CDL3BLACKCROWS,
    "CDL3INSIDE": talib.CDL3INSIDE,
    "CDL3LINESTRIKE": talib.CDL3LINESTRIKE,
    "CDL3OUTSIDE": talib.CDL3OUTSIDE,
    "CDL3STARSINSOUTH": talib.CDL3STARSINSOUTH,
    "CDL3WHITESOLDIERS": talib.CDL3WHITESOLDIERS,
    "CDLABANDONEDBABY": talib.CDLABANDONEDBABY,
    "CDLADVANCEBLOCK": talib.CDLADVANCEBLOCK,
    "CDLBELTHOLD": talib.CDLBELTHOLD,
    "CDLBREAKAWAY": talib.CDLBREAKAWAY,
    "CDLCLOSINGMARUBOZU": talib.CDLCLOSINGMARUBOZU,
    "CDLCONCEALBABYSWALL": talib.CDLCONCEALBABYSWALL,
    "CDLCOUNTERATTACK": talib.CDLCOUNTERATTACK,
    "CDLDARKCLOUDCOVER": talib.CDLDARKCLOUDCOVER,
    "CDLDOJI": talib.CDLDOJI,
    "CDLDOJISTAR": talib.CDLDOJISTAR,
    "CDLDRAGONFLYDOJI": talib.CDLDRAGONFLYDOJI,
    "CDLENGULFING": talib.CDLENGULFING,
    "CDLEVENINGDOJISTAR": talib.CDLEVENINGDOJISTAR,
    "CDLEVENINGSTAR": talib.CDLEVENINGSTAR,
    "CDLGAPSIDESIDEWHITE": talib.CDLGAPSIDESIDEWHITE,
    "CDLGRAVESTONEDOJI": talib.CDLGRAVESTONEDOJI,
    "CDLHAMMER": talib.CDLHAMMER,
    "CDLHANGINGMAN": talib.CDLHANGINGMAN,
    "CDLHARAMI": talib.CDLHARAMI,
    "CDLHARAMICROSS": talib.CDLHARAMICROSS,
    "CDLHIGHWAVE": talib.CDLHIGHWAVE,
    "CDLHIKKAKE": talib.CDLHIKKAKE,
    "CDLHIKKAKEMOD": talib.CDLHIKKAKEMOD,
    "CDLHOMINGPIGEON": talib.CDLHOMINGPIGEON,
    "CDLIDENTICAL3CROWS": talib.CDLIDENTICAL3CROWS,
    "CDLINNECK": talib.CDLINNECK,
    "CDLINVERTEDHAMMER": talib.CDLINVERTEDHAMMER,
    "CDLKICKING": talib.CDLKICKING,
    "CDLKICKINGBYLENGTH": talib.CDLKICKINGBYLENGTH,
    "CDLLADDERBOTTOM": talib.CDLLADDERBOTTOM,
    "CDLLONGLEGGEDDOJI": talib.CDLLONGLEGGEDDOJI,
    "CDLLONGLINE": talib.CDLLONGLINE,
    "CDLMARUBOZU": talib.CDLMARUBOZU,
    "CDLMATCHINGLOW": talib.CDLMATCHINGLOW,
    "CDLMATHOLD": talib.CDLMATHOLD,
    "CDLMORNINGDOJISTAR": talib.CDLMORNINGDOJISTAR,
    "CDLMORNINGSTAR": talib.CDLMORNINGSTAR,
    "CDLONNECK": talib.CDLONNECK,
    "CDLPIERCING": talib.CDLPIERCING,
    "CDLRICKSHAWMAN": talib.CDLRICKSHAWMAN,
    "CDLRISEFALL3METHODS": talib.CDLRISEFALL3METHODS,
    "CDLSEPARATINGLINES": talib.CDLSEPARATINGLINES,
    "CDLSHOOTINGSTAR": talib.CDLSHOOTINGSTAR,
    "CDLSHORTLINE": talib.CDLSHORTLINE,
    "CDLSPINNINGTOP": talib.CDLSPINNINGTOP,
    "CDLSTALLEDPATTERN": talib.CDLSTALLEDPATTERN,
    "CDLSTICKSANDWICH": talib.CDLSTICKSANDWICH,
    "CDLTAKURI": talib.CDLTAKURI,
    "CDLTASUKIGAP": talib.CDLTASUKIGAP,
    "CDLTHRUSTING": talib.CDLTHRUSTING,
    "CDLTRISTAR": talib.CDLTRISTAR,
    "CDLUNIQUE3RIVER": talib.CDLUNIQUE3RIVER,
    "CDLUPSIDEGAP2CROWS": talib.CDLUPSIDEGAP2CROWS,
    "CDLXSIDEGAP3METHODS": talib.CDLXSIDEGAP3METHODS,
}


@lru_cache(maxsize=1)
def list_patterns() -> list[str]:
    """List all available candlestick pattern names."""
    return sorted(_PATTERNS.keys())


def get_pattern_function(name: str) -> Callable | None:
    """Get a candlestick pattern function by name."""
    return _PATTERNS.get(name.upper())


@register_indicator("candlestick")
def compute_candlestick(
    df: pd.DataFrame,
    patterns: list[str] | None = None,
) -> pd.DataFrame:
    """Detect candlestick patterns in OHLCV data.

    Args:
        df: DataFrame with Open, High, Low, Close columns.
        patterns: List of TA-Lib pattern names to detect.
                  If None, detects all commonly used patterns:
                  Engulfing, Hammer, Shooting Star, Doji, Morning Star,
                  Evening Star, 3 White Soldiers, 3 Black Crows.

    Returns:
        DataFrame with pattern detection columns.
        Each pattern column contains: >0 for bullish, <0 for bearish, 0 for no pattern.
    """
    result = df.copy()

    open_p = df["Open"].values
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values

    if patterns is None:
        patterns = [
            "CDLENGULFING",
            "CDLHAMMER",
            "CDLSHOOTINGSTAR",
            "CDLDOJI",
            "CDLMORNINGSTAR",
            "CDLEVENINGSTAR",
            "CDL3WHITESOLDIERS",
            "CDL3BLACKCROWS",
            "CDLMORNINGDOJISTAR",
            "CDLEVENINGDOJISTAR",
            "CDLDRAGONFLYDOJI",
            "CDLGRAVESTONEDOJI",
            "CDLINVERTEDHAMMER",
            "CDLHANGINGMAN",
            "CDLPIERCING",
            "CDLDARKCLOUDCOVER",
            "CDLHARAMI",
            "CDLHARAMICROSS",
        ]

    for pattern_name in patterns:
        func = get_pattern_function(pattern_name)
        if func is not None:
            pattern_label = pattern_name.replace("CDL", "")
            result[f"Pattern_{pattern_label}"] = func(open_p, high, low, close)

    # Aggregate: any bullish or bearish pattern
    pattern_cols = [c for c in result.columns if c.startswith("Pattern_")]
    if pattern_cols:
        result["Pattern_Bullish"] = (result[pattern_cols] > 0).any(axis=1)
        result["Pattern_Bearish"] = (result[pattern_cols] < 0).any(axis=1)
        result["Pattern_Count"] = (result[pattern_cols] != 0).sum(axis=1)

    return result
