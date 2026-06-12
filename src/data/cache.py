"""SQLite cache helpers for OHLCV data."""

import json
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
from sqlalchemy import select, delete, and_
from sqlalchemy.ext.asyncio import AsyncSession

from src.data.database import OhlcvCacheModel, TickerModel, get_session_factory


def _normalize_ts(ts: datetime) -> datetime:
    """Ensure timestamp is timezone-naive for DB storage."""
    if ts.tzinfo is not None:
        ts = ts.replace(tzinfo=None)
    return ts


async def get_or_create_ticker(session: AsyncSession, symbol: str) -> TickerModel:
    """Get a ticker row by symbol, creating it if not exists."""
    stmt = select(TickerModel).where(TickerModel.symbol == symbol.upper())
    result = await session.execute(stmt)
    row = result.scalar_one_or_none()
    if row is None:
        row = TickerModel(symbol=symbol.upper())
        session.add(row)
        await session.flush()
    return row


async def cache_bars(
    session: AsyncSession,
    ticker: str,
    df: pd.DataFrame,
    interval: str = "1d",
) -> int:
    """Cache OHLCV bars from a DataFrame.

    Returns the number of bars inserted/updated.
    """
    if df.empty:
        return 0
    ticker_row = await get_or_create_ticker(session, ticker)
    count = 0
    for idx, row in df.iterrows():
        ts = idx if isinstance(idx, datetime) else pd.Timestamp(idx).to_pydatetime()
        ts = _normalize_ts(ts)
        # Upsert
        stmt = select(OhlcvCacheModel).where(
            and_(
                OhlcvCacheModel.ticker_id == ticker_row.id,
                OhlcvCacheModel.interval == interval,
                OhlcvCacheModel.timestamp == ts,
            )
        )
        res = await session.execute(stmt)
        existing = res.scalar_one_or_none()
        if existing:
            existing.open = float(row.get("Open", 0))
            existing.high = float(row.get("High", 0))
            existing.low = float(row.get("Low", 0))
            existing.close = float(row.get("Close", 0))
            existing.volume = float(row.get("Volume", 0))
            existing.fetched_at = datetime.utcnow()
        else:
            bar = OhlcvCacheModel(
                ticker_id=ticker_row.id,
                interval=interval,
                timestamp=ts,
                open=float(row.get("Open", 0)),
                high=float(row.get("High", 0)),
                low=float(row.get("Low", 0)),
                close=float(row.get("Close", 0)),
                volume=float(row.get("Volume", 0)),
                vwap=None,
                fetched_at=datetime.utcnow(),
            )
            session.add(bar)
        count += 1
    await session.flush()
    return count


async def get_cached_bars(
    session: AsyncSession,
    ticker: str,
    interval: str = "1d",
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int = 0,
) -> pd.DataFrame:
    """Retrieve cached OHLCV bars as a DataFrame."""
    ticker_row = await _get_ticker_row(session, ticker)
    if ticker_row is None:
        return pd.DataFrame()

    stmt = select(OhlcvCacheModel).where(
        and_(
            OhlcvCacheModel.ticker_id == ticker_row.id,
            OhlcvCacheModel.interval == interval,
        )
    )
    if start:
        stmt = stmt.where(OhlcvCacheModel.timestamp >= _normalize_ts(start))
    if end:
        stmt = stmt.where(OhlcvCacheModel.timestamp <= _normalize_ts(end))
    stmt = stmt.order_by(OhlcvCacheModel.timestamp.asc())
    if limit > 0:
        stmt = stmt.limit(limit)
    else:
        stmt = stmt.limit(2000)  # safety cap

    result = await session.execute(stmt)
    rows = result.scalars().all()

    if not rows:
        return pd.DataFrame()

    data = []
    for r in rows:
        data.append({
            "timestamp": pd.Timestamp(r.timestamp),
            "Open": r.open,
            "High": r.high,
            "Low": r.low,
            "Close": r.close,
            "Volume": r.volume,
        })
    df = pd.DataFrame(data)
    if not df.empty:
        df = df.set_index("timestamp")
        df = df.sort_index()
    return df


async def get_cache_gap(
    session: AsyncSession,
    ticker: str,
    interval: str = "1d",
    start: date | None = None,
    end: date | None = None,
) -> tuple[Optional[datetime], Optional[datetime]]:
    """Determine the missing date range in cache for a given ticker + interval.

    Returns (earliest_missing, latest_missing) or (None, None) if fully cached.
    """
    ticker_row = await _get_ticker_row(session, ticker)
    if ticker_row is None:
        return (None, None)

    stmt = select(OhlcvCacheModel).where(
        and_(
            OhlcvCacheModel.ticker_id == ticker_row.id,
            OhlcvCacheModel.interval == interval,
        )
    ).order_by(OhlcvCacheModel.timestamp.asc())

    result = await session.execute(stmt)
    rows = result.scalars().all()

    if not rows:
        return (
            datetime.combine(start or date.today() - timedelta(days=365), datetime.min.time()),
            datetime.combine(end or date.today(), datetime.min.time()),
        )

    existing_dates = {r.timestamp.date() for r in rows if r.timestamp is not None}
    start_d = start or (min(existing_dates) if existing_dates else date.today() - timedelta(days=365))
    end_d = end or date.today()

    missing_start: datetime | None = None
    missing_end: datetime | None = None

    for d in _daterange(start_d, end_d):
        if d not in existing_dates:
            if missing_start is None:
                missing_start = datetime.combine(d, datetime.min.time())
            missing_end = datetime.combine(d, datetime.min.time())

    return (missing_start, missing_end)


async def clear_stale_cache(session: AsyncSession, max_age_days: int = 1, interval: str = "5m") -> int:
    """Remove cached intraday bars older than max_age_days."""
    cutoff = datetime.utcnow() - timedelta(days=max_age_days)
    stmt = delete(OhlcvCacheModel).where(
        and_(
            OhlcvCacheModel.interval == interval,
            OhlcvCacheModel.timestamp < cutoff,
        )
    )
    result = await session.execute(stmt)
    await session.commit()
    return result.rowcount


async def _get_ticker_row(session: AsyncSession, symbol: str) -> Optional[TickerModel]:
    stmt = select(TickerModel).where(TickerModel.symbol == symbol.upper())
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


def _daterange(start: date, end: date):
    """Yield each date from start to end inclusive."""
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)
