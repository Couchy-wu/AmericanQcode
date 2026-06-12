"""Composite strategy: combine multiple strategies with AND/OR voting logic."""

import pandas as pd

from src.core.models import Signal, SignalDirection
from src.strategies.base import Strategy


class CompositeStrategy(Strategy):
    """Aggregate signals from multiple sub-strategies with configurable voting logic.

    AND mode: All sub-strategies must agree on direction.
    OR mode: Any sub-strategy's signal is emitted.
    WEIGHTED mode: Weighted voting across strategies.
    """

    name = "composite"
    required_indicators = []  # Depends on sub-strategies
    timeframe = "1d"
    min_confidence = 0.5

    MODE_AND = "and"
    MODE_OR = "or"
    MODE_WEIGHTED = "weighted"

    def __init__(
        self,
        strategies: list[Strategy],
        mode: str = "or",
        weights: dict[str, float] | None = None,
    ):
        """
        Args:
            strategies: List of Strategy instances.
            mode: 'and', 'or', or 'weighted'.
            weights: Per-strategy weights for weighted mode. Defaults to equal weight.
        """
        self.strategies = strategies
        self.mode = mode
        self.weights = weights or {}
        # Collect all required indicators from sub-strategies
        req = set()
        for s in strategies:
            req.update(s.required_indicators)
        self.required_indicators = list(req)

    def analyze(self, df: pd.DataFrame) -> list[Signal]:
        all_signals: list[Signal] = []

        # Collect signals from all sub-strategies
        strategy_signals: dict[str, list[Signal]] = {}
        for strat in self.strategies:
            try:
                sigs = strat.analyze(df)
                strategy_signals[strat.name] = sigs
                all_signals.extend(sigs)
            except Exception:
                continue

        if not all_signals:
            return []

        ticker = df.index.name or "UNKNOWN"
        price = float(df["Close"].iloc[-1]) if "Close" in df.columns else 0.0

        if self.mode == self.MODE_OR:
            # Return all signals as-is
            return all_signals

        elif self.mode == self.MODE_AND:
            # All strategies must agree on direction
            bullish_count = sum(1 for s in all_signals if s.direction == SignalDirection.BULLISH)
            bearish_count = sum(1 for s in all_signals if s.direction == SignalDirection.BEARISH)
            total_strats = len(self.strategies)

            results = []
            if bullish_count == total_strats:
                avg_conf = sum(s.confidence for s in all_signals if s.direction == SignalDirection.BULLISH) / total_strats
                results.append(self._make_signal(
                    ticker=ticker,
                    direction=SignalDirection.BULLISH,
                    confidence=min(avg_conf + 0.1, 1.0),
                    reasoning=f"Composite AND: {total_strats}/{total_strats} strategies bullish",
                    price=price,
                    indicators_used=self.required_indicators,
                ))
            if bearish_count == total_strats:
                avg_conf = sum(s.confidence for s in all_signals if s.direction == SignalDirection.BEARISH) / total_strats
                results.append(self._make_signal(
                    ticker=ticker,
                    direction=SignalDirection.BEARISH,
                    confidence=min(avg_conf + 0.1, 1.0),
                    reasoning=f"Composite AND: {total_strats}/{total_strats} strategies bearish",
                    price=price,
                    indicators_used=self.required_indicators,
                ))
            return results

        elif self.mode == self.MODE_WEIGHTED:
            # Weighted voting
            bullish_score = 0.0
            bearish_score = 0.0
            total_weight = 0.0

            for sig in all_signals:
                w = self.weights.get(sig.strategy, 1.0)
                if sig.direction == SignalDirection.BULLISH:
                    bullish_score += sig.confidence * w
                else:
                    bearish_score += sig.confidence * w
                total_weight += w

            results = []
            if total_weight > 0:
                norm_bullish = bullish_score / total_weight
                norm_bearish = bearish_score / total_weight

                if norm_bullish > 0.5:
                    results.append(self._make_signal(
                        ticker=ticker,
                        direction=SignalDirection.BULLISH,
                        confidence=min(norm_bullish, 1.0),
                        reasoning=f"Composite Weighted: bullish={norm_bullish:.2f}, bearish={norm_bearish:.2f}",
                        price=price,
                        indicators_used=self.required_indicators,
                    ))
                elif norm_bearish > 0.5:
                    results.append(self._make_signal(
                        ticker=ticker,
                        direction=SignalDirection.BEARISH,
                        confidence=min(norm_bearish, 1.0),
                        reasoning=f"Composite Weighted: bullish={norm_bullish:.2f}, bearish={norm_bearish:.2f}",
                        price=price,
                        indicators_used=self.required_indicators,
                    ))
            return results

        return all_signals
