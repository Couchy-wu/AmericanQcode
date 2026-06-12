"""Bollinger Bands squeeze breakout strategy.

A squeeze (narrowing bands) often precedes a sharp price breakout.
"""

import pandas as pd

from src.core.models import Signal, SignalDirection
from src.strategies.base import Strategy


class BollingerSqueezeStrategy(Strategy):
    """Detect Bollinger Band squeeze and subsequent breakout.

    - Squeeze + price breaks above middle band → BULLISH
    - Squeeze + price breaks below middle band → BEARISH
    - Price touches lower band and reverses → BULLISH reversal
    - Price touches upper band and reverses → BEARISH reversal
    """

    name = "bollinger_squeeze"
    required_indicators = ["bollinger"]
    timeframe = "1d"
    min_confidence = 0.6

    def __init__(
        self,
        period: int = 20,
        nbdev: float = 2.0,
        squeeze_threshold: float = 0.1,
    ):
        self.period = period
        self.nbdev = nbdev
        self.squeeze_threshold = squeeze_threshold

    def analyze(self, df: pd.DataFrame) -> list[Signal]:
        signals: list[Signal] = []

        required = ["BB_Upper", "BB_Middle", "BB_Lower", "Close"]
        for col in required:
            if col not in df.columns:
                return signals

        if len(df) < 5:
            return signals

        ticker = df.index.name or "UNKNOWN"
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        price = float(last_row["Close"])

        # Squeeze detection (low bandwidth)
        squeeze_col = "BB_Squeeze"
        is_squeeze = last_row.get(squeeze_col, False) if squeeze_col in df.columns else False

        # Breakout from squeeze: price was inside bands, now breaking
        if is_squeeze:
            # Bullish breakout above middle
            if last_row["Close"] > last_row["BB_Middle"] and prev_row["Close"] <= prev_row["BB_Middle"]:
                confidence = 0.7
                reasoning = "Bollinger Squeeze breakout: price broke above middle band after squeeze"

                if "Volume" in df.columns:
                    avg_vol = df["Volume"].rolling(window=20).mean().iloc[-1]
                    if last_row["Volume"] > avg_vol * 1.5:
                        confidence += 0.1
                        reasoning += " with volume spike"

                signals.append(self._make_signal(
                    ticker=ticker,
                    direction=SignalDirection.BULLISH,
                    confidence=confidence,
                    reasoning=reasoning,
                    price=price,
                    indicators_used=["bollinger"],
                ))

            # Bearish breakout below middle
            if last_row["Close"] < last_row["BB_Middle"] and prev_row["Close"] >= prev_row["BB_Middle"]:
                confidence = 0.65
                reasoning = "Bollinger Squeeze breakdown: price broke below middle band after squeeze"

                if "Volume" in df.columns:
                    avg_vol = df["Volume"].rolling(window=20).mean().iloc[-1]
                    if last_row["Volume"] > avg_vol * 1.5:
                        confidence += 0.1
                        reasoning += " with volume spike"

                signals.append(self._make_signal(
                    ticker=ticker,
                    direction=SignalDirection.BEARISH,
                    confidence=confidence,
                    reasoning=reasoning,
                    price=price,
                    indicators_used=["bollinger"],
                ))

        # Lower band touch + reversal (bullish reversal)
        if "BB_CrossLower" in df.columns:
            # Was below lower band, now above it
            if last_row.get("BB_CrossLower", False) and not prev_row.get("BB_CrossLower", True):
                pass  # Still touching
            if (prev_row["Close"] <= prev_row["BB_Lower"] and
                    last_row["Close"] > last_row["BB_Lower"]):
                signals.append(self._make_signal(
                    ticker=ticker,
                    direction=SignalDirection.BULLISH,
                    confidence=0.6,
                    reasoning="Bollinger Band reversal: price bounced off lower band",
                    price=price,
                    indicators_used=["bollinger"],
                ))

        # Upper band touch + reversal (bearish reversal)
        if (prev_row["Close"] >= prev_row["BB_Upper"] and
                last_row["Close"] < last_row["BB_Upper"]):
            signals.append(self._make_signal(
                ticker=ticker,
                direction=SignalDirection.BEARISH,
                confidence=0.55,
                reasoning="Bollinger Band reversal: price rejected at upper band",
                price=price,
                indicators_used=["bollinger"],
            ))

        return signals
