"""Signal pipeline: deduplication, filtering, ranking of trading signals."""

from datetime import datetime, timedelta

from loguru import logger

from src.core.models import Signal, SignalDirection


class SignalPipeline:
    """Post-processing pipeline for raw strategy signals.

    Steps:
        1. Deduplicate: remove duplicate signals for same ticker+strategy within window.
        2. Filter: remove signals below minimum confidence.
        3. Rank: sort by confidence descending.
        4. Cap: limit total signals per scan.
    """

    def __init__(
        self,
        min_confidence: float = 0.5,
        dedup_window_bars: int = 3,
        max_signals: int = 50,
        max_per_ticker: int = 5,
    ):
        self.min_confidence = min_confidence
        self.dedup_window_bars = dedup_window_bars
        self.max_signals = max_signals
        self.max_per_ticker = max_per_ticker

    def process(
        self,
        signals: list[Signal],
        now: datetime | None = None,
    ) -> list[Signal]:
        """Run the signal pipeline.

        Args:
            signals: Raw list of Signal objects.
            now: Current time for expiration check.

        Returns:
            Filtered and ranked list of Signal objects.
        """
        if not signals:
            return []

        now = now or datetime.now()

        # Step 1: Deduplicate
        deduped = self._deduplicate(signals)

        # Step 2: Filter by confidence
        filtered = [s for s in deduped if s.confidence >= self.min_confidence]

        # Step 3: Filter expired
        filtered = [s for s in filtered if not s.is_expired(now)]

        # Step 4: Rank by confidence (descending)
        ranked = sorted(filtered, key=lambda s: s.confidence, reverse=True)

        # Step 5: Cap per ticker
        capped: list[Signal] = []
        ticker_counts: dict[str, int] = {}
        for sig in ranked:
            count = ticker_counts.get(sig.ticker, 0)
            if count >= self.max_per_ticker:
                continue
            capped.append(sig)
            ticker_counts[sig.ticker] = count + 1

        # Step 6: Cap total
        result = capped[:self.max_signals]

        return result

    def _deduplicate(self, signals: list[Signal]) -> list[Signal]:
        """Remove duplicate signals for the same ticker+strategy+timeframe.

        Within the dedup window, only the highest-confidence signal is kept
        for each (ticker, strategy) combination.
        """
        # Group by (ticker, strategy)
        groups: dict[tuple[str, str], list[Signal]] = {}
        for sig in signals:
            key = (sig.ticker.upper(), sig.strategy)
            if key not in groups:
                groups[key] = []
            groups[key].append(sig)

        # For each group, keep only the highest confidence signal
        result: list[Signal] = []
        for key, sigs in groups.items():
            best = max(sigs, key=lambda s: s.confidence)
            result.append(best)

        return result
