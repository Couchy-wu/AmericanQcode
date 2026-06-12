"""Moving Average crossover and breakout strategy."""

import pandas as pd

from src.core.models import Signal, SignalDirection
from src.strategies.base import Strategy


class MABreakoutStrategy(Strategy):
    """Golden Cross / Death Cross of fast and slow Moving Averages.

    - Fast MA crosses above Slow MA → BULLISH (Golden Cross)
    - Fast MA crosses below Slow MA → BEARISH (Death Cross)
    """

    name = "ma_breakout"
    required_indicators = ["ma_cross"]
    timeframe = "1d"
    min_confidence = 0.55

    def __init__(
        self,
        fast_ma: int = 20,
        slow_ma: int = 50,
        volume_factor: float = 1.5,
    ):
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.volume_factor = volume_factor

    def analyze(self, df: pd.DataFrame) -> list[Signal]:
        signals: list[Signal] = []

        required = ["MA_GoldenCross", "MA_DeathCross", "Close"]
        for col in required:
            if col not in df.columns:
                return signals

        ticker = df.index.name or "UNKNOWN"
        last_row = df.iloc[-1]
        price = float(last_row["Close"])

        # Golden Cross
        if last_row.get("MA_GoldenCross", False):
            confidence = 0.55
            reasoning = f"MA Golden Cross: MA{self.fast_ma} crossed above MA{self.slow_ma}"

            # Volume confirmation
            if "Volume" in df.columns:
                avg_vol = df["Volume"].rolling(window=20).mean().iloc[-1]
                if last_row["Volume"] > avg_vol * self.volume_factor:
                    confidence += 0.15
                    reasoning += " with volume spike"

            # Trend confirmation from ADX
            if "ADX_Trending" in df.columns and last_row.get("ADX_Trending", False):
                confidence += 0.1
                reasoning += "; ADX confirms trend"

            if confidence >= self.min_confidence:
                signals.append(self._make_signal(
                    ticker=ticker,
                    direction=SignalDirection.BULLISH,
                    confidence=confidence,
                    reasoning=reasoning,
                    price=price,
                ))

        # Death Cross
        if last_row.get("MA_DeathCross", False):
            confidence = 0.55
            reasoning = f"MA Death Cross: MA{self.fast_ma} crossed below MA{self.slow_ma}"

            if "Volume" in df.columns:
                avg_vol = df["Volume"].rolling(window=20).mean().iloc[-1]
                if last_row["Volume"] > avg_vol * self.volume_factor:
                    confidence += 0.1
                    reasoning += " with volume spike"

            if "ADX_Trending" in df.columns and last_row.get("ADX_Trending", False):
                confidence += 0.1
                reasoning += "; ADX confirms trend"

            if confidence >= self.min_confidence:
                signals.append(self._make_signal(
                    ticker=ticker,
                    direction=SignalDirection.BEARISH,
                    confidence=confidence,
                    reasoning=reasoning,
                    price=price,
                ))

        # Price crossing MA
        fast_ma_col = f"MA_{self.fast_ma}"
        if fast_ma_col in df.columns:
            close = df["Close"]
            ma = df[fast_ma_col]
            # Price crossing above MA
            if len(close) >= 2 and len(ma) >= 2:
                if close.iloc[-1] > ma.iloc[-1] and close.iloc[-2] <= ma.iloc[-2]:
                    if not last_row.get("MA_GoldenCross", False):  # Don't duplicate
                        signals.append(self._make_signal(
                            ticker=ticker,
                            direction=SignalDirection.BULLISH,
                            confidence=0.5,
                            reasoning=f"Price broke above MA{self.fast_ma}",
                            price=price,
                        ))

        return signals
