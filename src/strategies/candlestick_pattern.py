"""Candlestick pattern-based strategy."""

import pandas as pd

from src.core.models import Signal, SignalDirection
from src.strategies.base import Strategy


class CandlestickPatternStrategy(Strategy):
    """Generate signals from recognized candlestick patterns.

    Bullish patterns: Hammer, Engulfing (bullish), Morning Star, Piercing, etc.
    Bearish patterns: Shooting Star, Engulfing (bearish), Evening Star, Dark Cloud Cover, etc.
    """

    name = "candlestick_pattern"
    required_indicators = ["candlestick"]
    timeframe = "1d"
    min_confidence = 0.5

    # Pattern classification
    BULLISH_PATTERNS = {
        "HAMMER": 0.55,
        "INVERTEDHAMMER": 0.55,
        "ENGULFING": 0.65,         # TA-Lib returns positive for bullish engulfing
        "MORNINGSTAR": 0.70,
        "MORNINGDOJISTAR": 0.75,
        "PIERCING": 0.60,
        "3WHITESOLDIERS": 0.65,
        "DRAGONFLYDOJI": 0.55,
        "HARAMI": 0.55,            # TA-Lib returns positive for bullish harami
        "HARAMICROSS": 0.55,
        "HOMINGPIGEON": 0.50,
    }

    BEARISH_PATTERNS = {
        "SHOOTINGSTAR": 0.55,
        "ENGULFING": 0.65,         # TA-Lib returns negative for bearish engulfing
        "EVENINGSTAR": 0.70,
        "EVENINGDOJISTAR": 0.75,
        "DARKCLOUDCOVER": 0.60,
        "3BLACKCROWS": 0.65,
        "GRAVESTONEDOJI": 0.55,
        "HANGINGMAN": 0.55,
        "HARAMI": 0.55,            # TA-Lib returns negative for bearish harami
    }

    def analyze(self, df: pd.DataFrame) -> list[Signal]:
        signals: list[Signal] = []

        if "Close" not in df.columns:
            return signals

        ticker = df.index.name or "UNKNOWN"
        last_row = df.iloc[-1]
        price = float(last_row["Close"])

        pattern_cols = [c for c in df.columns if c.startswith("Pattern_")]
        if not pattern_cols:
            return signals

        for col in pattern_cols:
            pattern_name = col.replace("Pattern_", "")
            val = last_row.get(col, 0)
            if pd.isna(val) or val == 0:
                continue

            # Positive = bullish, Negative = bearish
            if val > 0 and pattern_name in self.BULLISH_PATTERNS:
                base_confidence = self.BULLISH_PATTERNS[pattern_name]
                signals.append(self._make_signal(
                    ticker=ticker,
                    direction=SignalDirection.BULLISH,
                    confidence=base_confidence,
                    reasoning=f"Bullish candlestick pattern: {pattern_name}",
                    price=price,
                    indicators_used=["candlestick"],
                ))
            elif val < 0 and pattern_name in self.BEARISH_PATTERNS:
                base_confidence = self.BEARISH_PATTERNS[pattern_name]
                signals.append(self._make_signal(
                    ticker=ticker,
                    direction=SignalDirection.BEARISH,
                    confidence=base_confidence,
                    reasoning=f"Bearish candlestick pattern: {pattern_name}",
                    price=price,
                    indicators_used=["candlestick"],
                ))

        return signals
