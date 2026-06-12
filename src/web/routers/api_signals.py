"""REST API for signals."""

from datetime import datetime

from fastapi import APIRouter, Query

from src.data.database import get_session_factory
from src.data.repository import SignalRepository
from src.core.models import SignalDirection

router = APIRouter()


@router.get("/signals")
async def get_signals(
    ticker: str | None = Query(None),
    strategy: str | None = Query(None),
    direction: str | None = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """Get signals with optional filters."""
    dir_enum = None
    if direction:
        try:
            dir_enum = SignalDirection(direction.upper())
        except ValueError:
            pass

    factory = get_session_factory()
    async with factory() as session:
        repo = SignalRepository(session)
        signals = await repo.get_signals(
            ticker=ticker, strategy=strategy, direction=dir_enum,
            limit=limit, offset=offset,
        )

    return [
        {
            "ticker": s.ticker,
            "timestamp": s.timestamp.isoformat() if s.timestamp else None,
            "direction": s.direction.value,
            "confidence": s.confidence,
            "strategy": s.strategy,
            "reasoning": s.reasoning,
            "price": s.price_at_signal,
        }
        for s in signals
    ]
