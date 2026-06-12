"""RSI-based strategy: oversold/overbought crosses and divergence."""

import pandas as pd

from src.core.models import Signal, SignalDirection
from src.strategies.base import Strategy


class RSIDivergenceStrategy(Strategy):
    """Generate signals based on RSI conditions.

    - RSI exiting oversold → BULLISH
    - RSI exiting overbought → BEARISH
    - RSI Bullish Divergence → BULLISH
    - RSI Bearish Divergence → BEARISH
    """

    name = "rsi_divergence"
    required_indicators = ["rsi"]
    timeframe = "1d"
    min_confidence = 0.65

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        divergence_window: int = 20,
    ):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.divergence_window = divergence_window

    def analyze(self, df: pd.DataFrame) -> list[Signal]:
        signals: list[Signal] = []

        required = ["RSI", "Close"]
        for col in required:
            if col not in df.columns:
                return signals

        ticker = df.index.name or "UNKNOWN"
        last_row = df.iloc[-1]
        price = float(last_row["Close"])

        # RSI exiting oversold (bullish)
        if last_row.get("RSI_ExitOversold", False):
            rsi_val = float(last_row["RSI"])
            confidence = 0.65
            reasoning = f"RSI exited oversold zone at {rsi_val:.1f}" + \
                        f" (crossed above {self.oversold})"
            signals.append(self._make_signal(
                ticker=ticker,
                direction=SignalDirection.BULLISH,
                confidence=confidence,
                reasoning=reasoning,
                price=price,
                indicators_used=["rsi"],
            ))

        # RSI exiting overbought (bearish)
        if last_row.get("RSI_ExitOverbought", False):
            rsi_val = float(last_row["RSI"])
            confidence = 0.65
            reasoning = f"RSI exited overbought zone at {rsi_val:.1f}" + \
                        f" (crossed below {self.overbought})"
            signals.append(self._make_signal(
                ticker=ticker,
                direction=SignalDirection.BEARISH,
                confidence=confidence,
                reasoning=reasoning,
                price=price,
                indicators_used=["rsi"],
            ))

        # RSI Divergence
        if last_row.get("RSI_BullishDiv", False):
            signals.append(self._make_signal(
                ticker=ticker,
                direction=SignalDirection.BULLISH,
                confidence=0.75,
                reasoning="RSI Bullish Divergence: price lower low, RSI higher low",
                price=price,
                indicators_used=["rsi"],
            ))

        if last_row.get("RSI_BearishDiv", False):
            signals.append(self._make_signal(
                ticker=ticker,
                direction=SignalDirection.BEARISH,
                confidence=0.75,
                reasoning="RSI Bearish Divergence: price higher high, RSI lower high",
                price=price,
                indicators_used=["rsi"],
            ))

        return signals
