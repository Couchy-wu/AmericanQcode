"""SQLAlchemy async engine, session, and ORM models."""

from datetime import datetime

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, UniqueConstraint,
    create_engine,
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, relationship

from src.core.config import get_database_url


# ─── Engine and Session ───────────────────────────────────────────────────────


def _build_url() -> str:
    url = get_database_url()
    # Ensure data directory exists
    import os
    from pathlib import Path

    if url.startswith("sqlite"):
        db_path = url.split("///")[-1]
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return url


_async_engine = None
_async_session_factory = None


def _get_engine():
    global _async_engine
    if _async_engine is None:
        _async_engine = create_async_engine(_build_url(), echo=False)
    return _async_engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _async_session_factory
    if _async_session_factory is None:
        _async_session_factory = async_sessionmaker(_get_engine(), class_=AsyncSession, expire_on_commit=False)
    return _async_session_factory


async def get_session() -> AsyncSession:
    """Yield a new async session. Use as FastAPI dependency."""
    factory = get_session_factory()
    async with factory() as session:
        yield session


async def init_db():
    """Create all tables."""
    engine = _get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# ─── ORM Base ─────────────────────────────────────────────────────────────────


class Base(DeclarativeBase):
    pass


# ─── Models ───────────────────────────────────────────────────────────────────


class TickerModel(Base):
    __tablename__ = "tickers"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=True)
    exchange = Column(String(10), nullable=True)   # NYSE / NASDAQ
    sector = Column(String(100), nullable=True)
    active = Column(Boolean, default=True)

    ohlcv_bars = relationship("OhlcvCacheModel", back_populates="ticker", cascade="all, delete-orphan")
    signals = relationship("SignalModel", back_populates="ticker", cascade="all, delete-orphan")


class OhlcvCacheModel(Base):
    __tablename__ = "ohlcv_cache"
    __table_args__ = (
        UniqueConstraint("ticker_id", "interval", "timestamp", name="uq_ohlcv_ticker_interval_ts"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker_id = Column(Integer, ForeignKey("tickers.id"), nullable=False)
    interval = Column(String(5), nullable=False, default="1d")
    timestamp = Column(DateTime, nullable=False)
    open = Column(Float, nullable=True)
    high = Column(Float, nullable=True)
    low = Column(Float, nullable=True)
    close = Column(Float, nullable=True)
    volume = Column(Float, nullable=True)
    vwap = Column(Float, nullable=True)
    fetched_at = Column(DateTime, default=datetime.utcnow)

    ticker = relationship("TickerModel", back_populates="ohlcv_bars")


class SignalModel(Base):
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker_id = Column(Integer, ForeignKey("tickers.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    direction = Column(String(10), nullable=False)   # BULLISH / BEARISH
    confidence = Column(Float, default=0.0)
    strategy = Column(String(50), nullable=False, index=True)
    indicators = Column(String, nullable=True)        # JSON array
    reasoning = Column(Text, nullable=True)
    price = Column(Float, nullable=True)
    expiration = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    ticker = relationship("TickerModel", back_populates="signals")


class WatchlistModel(Base):
    __tablename__ = "watchlists"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False)
    tickers = Column(Text, nullable=False)  # JSON array of symbols


class BacktestRunModel(Base):
    __tablename__ = "backtest_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy = Column(String(50), nullable=False)
    ticker = Column(String(10), nullable=False)
    date_range = Column(String(50), nullable=False)
    params = Column(Text, nullable=True)      # JSON config snapshot
    metrics = Column(Text, nullable=True)     # JSON: sharpe, max_dd, etc.
    created_at = Column(DateTime, default=datetime.utcnow)

    trades = relationship("BacktestTradeModel", back_populates="run", cascade="all, delete-orphan")


class BacktestTradeModel(Base):
    __tablename__ = "backtest_trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("backtest_runs.id"), nullable=False)
    entry_time = Column(DateTime, nullable=False)
    exit_time = Column(DateTime, nullable=False)
    direction = Column(String(5), nullable=False)   # BUY / SELL
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    pnl = Column(Float, nullable=False)
    pnl_pct = Column(Float, nullable=False)

    run = relationship("BacktestRunModel", back_populates="trades")
