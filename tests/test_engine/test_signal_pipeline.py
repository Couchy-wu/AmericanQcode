"""Tests for signal pipeline."""

from datetime import datetime, timedelta

from src.core.models import Signal, SignalDirection
from src.engine.signal_pipeline import SignalPipeline


def make_signal(ticker="AAPL", direction=SignalDirection.BULLISH, confidence=0.7, strategy="test"):
    """Helper to create a test Signal."""
    return Signal(
        ticker=ticker,
        timestamp=datetime.now(),
        direction=direction,
        confidence=confidence,
        strategy=strategy,
        indicators_used=["test"],
        reasoning="test signal",
        price_at_signal=150.0,
        expiration=datetime.now() + timedelta(days=5),
    )


def test_pipeline_filters_by_confidence():
    """Low-confidence signals should be removed."""
    pipeline = SignalPipeline(min_confidence=0.6)
    signals = [
        make_signal(confidence=0.8),
        make_signal(confidence=0.4),
        make_signal(confidence=0.9),
    ]
    result = pipeline.process(signals)
    assert len(result) == 2
    assert all(s.confidence >= 0.6 for s in result)


def test_pipeline_deduplicate():
    """Duplicate ticker+strategy combos should be deduped (keep highest conf)."""
    pipeline = SignalPipeline()
    signals = [
        make_signal(ticker="AAPL", strategy="macd_cross", confidence=0.6),
        make_signal(ticker="AAPL", strategy="macd_cross", confidence=0.9),
        make_signal(ticker="MSFT", strategy="macd_cross", confidence=0.7),
    ]
    result = pipeline.process(signals)
    assert len(result) == 2
    aapl_sig = [s for s in result if s.ticker == "AAPL"][0]
    assert aapl_sig.confidence == 0.9


def test_pipeline_caps_total():
    """Should cap at max_signals."""
    pipeline = SignalPipeline(max_signals=3)
    signals = [make_signal(ticker=f"TICK{i}", confidence=0.5 + i * 0.01) for i in range(10)]
    result = pipeline.process(signals)
    assert len(result) <= 3


def test_pipeline_ranks_by_confidence():
    """Signals should be sorted by confidence descending."""
    pipeline = SignalPipeline()
    signals = [
        make_signal(ticker="A", confidence=0.5),
        make_signal(ticker="B", confidence=0.9),
        make_signal(ticker="C", confidence=0.7),
    ]
    result = pipeline.process(signals)
    assert result[0].confidence >= result[1].confidence >= result[2].confidence


def test_empty_signals():
    """Empty input should return empty output."""
    pipeline = SignalPipeline()
    assert pipeline.process([]) == []
