"""High-level async repository for CRUD operations on signals, watchlists, etc."""

import json
from datetime import datetime
from typing import Optional

from sqlalchemy import select, delete, and_
from sqlalchemy.ext.asyncio import AsyncSession

from src.data.database import (
    SignalModel, WatchlistModel, BacktestRunModel, BacktestTradeModel,
    TickerModel,
)
from src.core.models import Signal, SignalDirection


class SignalRepository:
    """CRUD operations for trading signals."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def _get_ticker_id(self, symbol: str) -> int:
        stmt = select(TickerModel).where(TickerModel.symbol == symbol.upper())
        result = await self.session.execute(stmt)
        row = result.scalar_one_or_none()
        if row is None:
            row = TickerModel(symbol=symbol.upper())
            self.session.add(row)
            await self.session.flush()
        return row.id

    async def save_signal(self, signal: Signal) -> int:
        """Persist a Signal to the database. Returns the signal ID."""
        ticker_id = await self._get_ticker_id(signal.ticker)
        model = SignalModel(
            ticker_id=ticker_id,
            timestamp=signal.timestamp,
            direction=signal.direction.value,
            confidence=signal.confidence,
            strategy=signal.strategy,
            indicators=json.dumps(signal.indicators_used),
            reasoning=signal.reasoning,
            price=signal.price_at_signal,
            expiration=signal.expiration,
            created_at=datetime.utcnow(),
        )
        self.session.add(model)
        await self.session.commit()
        return model.id

    async def save_signals(self, signals: list[Signal]) -> int:
        """Batch persist signals. Returns count saved."""
        count = 0
        for sig in signals:
            await self.save_signal(sig)
            count += 1
        return count

    async def get_signals(
        self,
        ticker: str | None = None,
        strategy: str | None = None,
        direction: SignalDirection | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Signal]:
        """Query signals with optional filters."""
        stmt = select(SignalModel)

        if ticker:
            ticker_id = await self._get_ticker_id(ticker)
            stmt = stmt.where(SignalModel.ticker_id == ticker_id)
        if strategy:
            stmt = stmt.where(SignalModel.strategy == strategy)
        if direction:
            stmt = stmt.where(SignalModel.direction == direction.value)
        if start:
            stmt = stmt.where(SignalModel.timestamp >= start)
        if end:
            stmt = stmt.where(SignalModel.timestamp <= end)

        stmt = stmt.order_by(SignalModel.timestamp.desc()).offset(offset).limit(limit)
        result = await self.session.execute(stmt)
        rows = result.scalars().all()

        return [_model_to_signal(r) for r in rows]

    async def get_recent_signal_for_ticker(
        self, ticker: str, strategy: str, within_bars: int = 3,
    ) -> Optional[SignalModel]:
        """Check if a ticker+strategy combo already has a signal recently."""
        ticker_id = await self._get_ticker_id(ticker)
        cutoff = datetime.utcnow()  # Approximate — caller should refine if needed
        stmt = (
            select(SignalModel)
            .where(
                and_(
                    SignalModel.ticker_id == ticker_id,
                    SignalModel.strategy == strategy,
                )
            )
            .order_by(SignalModel.created_at.desc())
            .limit(1)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def expire_signals(self, before: datetime) -> int:
        """Delete signals older than the given cutoff."""
        stmt = delete(SignalModel).where(SignalModel.created_at < before)
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.rowcount


class WatchlistRepository:
    """CRUD operations for watchlists."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_watchlist(self, name: str) -> Optional[list[str]]:
        stmt = select(WatchlistModel).where(WatchlistModel.name == name)
        result = await self.session.execute(stmt)
        row = result.scalar_one_or_none()
        if row is None:
            return None
        return json.loads(row.tickers)

    async def save_watchlist(self, name: str, tickers: list[str]) -> None:
        stmt = select(WatchlistModel).where(WatchlistModel.name == name)
        result = await self.session.execute(stmt)
        row = result.scalar_one_or_none()
        if row:
            row.tickers = json.dumps([t.upper() for t in tickers])
        else:
            row = WatchlistModel(name=name, tickers=json.dumps([t.upper() for t in tickers]))
            self.session.add(row)
        await self.session.commit()

    async def delete_watchlist(self, name: str) -> bool:
        stmt = delete(WatchlistModel).where(WatchlistModel.name == name)
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.rowcount > 0

    async def list_watchlists(self) -> list[str]:
        stmt = select(WatchlistModel.name)
        result = await self.session.execute(stmt)
        return [r for r in result.scalars().all()]


class BacktestRepository:
    """CRUD operations for backtest results."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def save_run(
        self,
        strategy: str,
        ticker: str,
        date_range: str,
        params: dict,
        metrics: dict,
        trades: list[dict],
    ) -> int:
        run = BacktestRunModel(
            strategy=strategy,
            ticker=ticker,
            date_range=date_range,
            params=json.dumps(params),
            metrics=json.dumps(metrics),
        )
        self.session.add(run)
        await self.session.flush()

        for t in trades:
            trade = BacktestTradeModel(
                run_id=run.id,
                entry_time=t["entry_time"],
                exit_time=t["exit_time"],
                direction=t["direction"],
                entry_price=t["entry_price"],
                exit_price=t["exit_price"],
                quantity=t["quantity"],
                pnl=t["pnl"],
                pnl_pct=t["pnl_pct"],
            )
            self.session.add(trade)

        await self.session.commit()
        return run.id

    async def get_runs(self, limit: int = 20) -> list[dict]:
        stmt = select(BacktestRunModel).order_by(BacktestRunModel.created_at.desc()).limit(limit)
        result = await self.session.execute(stmt)
        rows = result.scalars().all()
        return [
            {
                "id": r.id,
                "strategy": r.strategy,
                "ticker": r.ticker,
                "date_range": r.date_range,
                "params": json.loads(r.params or "{}"),
                "metrics": json.loads(r.metrics or "{}"),
                "created_at": r.created_at,
            }
            for r in rows
        ]


def _model_to_signal(m: SignalModel) -> Signal:
    return Signal(
        ticker="",  # Will be resolved via join or separate lookup
        timestamp=m.timestamp,
        direction=SignalDirection(m.direction),
        confidence=m.confidence,
        strategy=m.strategy,
        indicators_used=json.loads(m.indicators) if m.indicators else [],
        reasoning=m.reasoning or "",
        price_at_signal=m.price or 0.0,
        expiration=m.expiration,
    )
