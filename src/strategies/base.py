"""Abstract base class for all trading strategies."""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta

import pandas as pd

from src.core.models import Signal, SignalDirection


class Strategy(ABC):
    """Base class for all trading strategies.

    Subclasses must define:
        name: Unique strategy name.
        required_indicators: List of indicator names needed.
        timeframe: Default bar interval.
        min_confidence: Default minimum confidence threshold.
        analyze(df) -> list[Signal]: Core analysis method.
    """

    name: str = "base"
    required_indicators: list[str] = []
    timeframe: str = "1d"
    min_confidence: float = 0.5

    @abstractmethod
    def analyze(self, df: pd.DataFrame) -> list[Signal]:
        """Analyze a DataFrame and generate trading signals.

        Args:
            df: DataFrame with OHLCV data and required indicator columns.

        Returns:
            List of Signal objects.
        """
        ...

    def _make_signal(
        self,
        ticker: str,
        direction: SignalDirection,
        confidence: float,
        reasoning: str,
        price: float,
        timestamp: datetime | None = None,
        indicators_used: list[str] | None = None,
        expiration_bars: int = 5,
    ) -> Signal:
        """Create a Signal object with default values from this strategy.

        Args:
            ticker: Stock symbol.
            direction: BULLISH or BEARISH.
            confidence: 0.0 to 1.0 confidence score.
            reasoning: Human-readable explanation.
            price: Current price.
            timestamp: Signal timestamp (defaults to now).
            indicators_used: List of indicator names used.
            expiration_bars: Bars until signal expires.

        Returns:
            Signal object.
        """
        ts = timestamp or datetime.now()
        expiration: datetime | None = None
        if self.timeframe == "1d":
            expiration = ts + timedelta(days=expiration_bars)
        elif self.timeframe == "1h":
            expiration = ts + timedelta(hours=expiration_bars * 1)
        else:
            expiration = ts + timedelta(hours=expiration_bars)

        return Signal(
            ticker=ticker.upper(),
            timestamp=ts,
            direction=direction,
            confidence=min(confidence, 1.0),
            strategy=self.name,
            indicators_used=indicators_used or self.required_indicators,
            reasoning=reasoning,
            price_at_signal=price,
            expiration=expiration,
        )

    def __repr__(self) -> str:
        return f"Strategy({self.name})"
