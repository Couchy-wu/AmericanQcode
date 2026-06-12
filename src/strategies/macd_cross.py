"""MACD Golden/Death Cross strategy."""

import pandas as pd

from src.core.models import Signal, SignalDirection
from src.strategies.base import Strategy


class MACDCrossStrategy(Strategy):
    """Generate signals when MACD line crosses the Signal line.

    Golden Cross (MACD crosses above Signal) → BULLISH
    Death Cross (MACD crosses below Signal) → BEARISH
    """

    name = "macd_cross"
    required_indicators = ["macd"]
    timeframe = "1d"
    min_confidence = 0.6

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        require_volume_confirmation: bool = True,
        volume_factor: float = 1.5,
    ):
        self.fast = fast
        self.slow = slow
        self.signal_period = signal
        self.require_volume_confirmation = require_volume_confirmation
        self.volume_factor = volume_factor

    def analyze(self, df: pd.DataFrame) -> list[Signal]:
        signals: list[Signal] = []

        required_cols = ["MACD_GoldenCross", "MACD_DeathCross", "Close"]
        if self.require_volume_confirmation:
            required_cols.append("Volume")

        for col in required_cols:
            if col not in df.columns:
                return signals

        ticker = df.index.name or "UNKNOWN"
        last_row = df.iloc[-1]
        price = float(last_row["Close"])

        # Check for Golden Cross on the last bar
        if last_row.get("MACD_GoldenCross", False):
            confidence = 0.6
            reasoning = f"MACD Golden Cross: MACD line crossed above Signal line"

            # Volume confirmation
            if self.require_volume_confirmation and "Volume" in df.columns:
                avg_vol = df["Volume"].rolling(window=20).mean().iloc[-1]
                if last_row["Volume"] > avg_vol * self.volume_factor:
                    confidence += 0.15
                    reasoning += " with volume confirmation"
                else:
                    confidence -= 0.1
                    reasoning += " (below avg volume)"

            # Additional: check if in oversold RSI zone (if RSI column present)
            if "RSI" in df.columns:
                rsi_val = df["RSI"].iloc[-1]
                if rsi_val < 30:
                    confidence += 0.1
                    reasoning += f"; RSI oversold ({rsi_val:.1f})"

            if confidence >= self.min_confidence:
                signals.append(
                    self._make_signal(
                        ticker=ticker,
                        direction=SignalDirection.BULLISH,
                        confidence=confidence,
                        reasoning=reasoning,
                        price=price,
                        indicators_used=["macd"] + (["rsi"] if "RSI" in df.columns else []),
                    )
                )

        # Check for Death Cross on the last bar
        if last_row.get("MACD_DeathCross", False):
            confidence = 0.6
            reasoning = f"MACD Death Cross: MACD line crossed below Signal line"

            if self.require_volume_confirmation and "Volume" in df.columns:
                avg_vol = df["Volume"].rolling(window=20).mean().iloc[-1]
                if last_row["Volume"] > avg_vol * self.volume_factor:
                    confidence += 0.1
                    reasoning += " with volume confirmation"

            if "RSI" in df.columns:
                rsi_val = df["RSI"].iloc[-1]
                if rsi_val > 70:
                    confidence += 0.1
                    reasoning += f"; RSI overbought ({rsi_val:.1f})"

            if confidence >= self.min_confidence:
                signals.append(
                    self._make_signal(
                        ticker=ticker,
                        direction=SignalDirection.BEARISH,
                        confidence=confidence,
                        reasoning=reasoning,
                        price=price,
                        indicators_used=["macd"] + (["rsi"] if "RSI" in df.columns else []),
                    )
                )

        # MACD Divergence signals
        if last_row.get("MACD_BullishDiv", False):
            signals.append(
                self._make_signal(
                    ticker=ticker,
                    direction=SignalDirection.BULLISH,
                    confidence=0.75,
                    reasoning="MACD Bullish Divergence: price made lower low but MACD made higher low",
                    price=price,
                    indicators_used=["macd"],
                )
            )

        if last_row.get("MACD_BearishDiv", False):
            signals.append(
                self._make_signal(
                    ticker=ticker,
                    direction=SignalDirection.BEARISH,
                    confidence=0.75,
                    reasoning="MACD Bearish Divergence: price made higher high but MACD made lower high",
                    price=price,
                    indicators_used=["macd"],
                )
            )

        return signals
