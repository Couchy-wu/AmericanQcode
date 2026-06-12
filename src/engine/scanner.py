"""Market scanner: orchestrates data fetching, indicator computation, and strategy analysis."""

import asyncio
from datetime import datetime

import pandas as pd
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.models import Signal
from src.data.base import AbstractDataProvider
from src.data.cache import cache_bars, get_cached_bars
from src.data.database import get_session_factory
from src.data.repository import SignalRepository
from src.engine.signal_pipeline import SignalPipeline
from src.indicators.base import list_indicators
from src.strategies.base import Strategy


class Scanner:
    """Scans a watchlist of tickers using configured strategies.

    Flow:
        1. Fetch OHLCV data for all tickers (from cache + provider).
        2. Compute required indicators for each ticker.
        3. Run each strategy on each ticker.
        4. Run signals through the signal pipeline.
        5. Persist signals to the database.
    """

    def __init__(
        self,
        provider: AbstractDataProvider,
        strategies: list[Strategy],
        watchlist: list[str],
        pipeline: SignalPipeline | None = None,
        enable_db: bool = False,
    ):
        self.provider = provider
        self.strategies = strategies
        self.watchlist = [t.upper() for t in watchlist]
        self.pipeline = pipeline or SignalPipeline()
        self.enable_db = enable_db

    async def scan(self) -> list[Signal]:
        """Run a full scan cycle. Returns all generated signals."""
        logger.info(f"Scanner: scanning {len(self.watchlist)} tickers with {len(self.strategies)} strategies")
        all_signals: list[Signal] = []

        for ticker in self.watchlist:
            try:
                signals = await self._scan_ticker(ticker)
                all_signals.extend(signals)
            except Exception as e:
                logger.error(f"Scanner: error scanning {ticker}: {e}")
                continue

        # Run through pipeline
        filtered = self.pipeline.process(all_signals)
        logger.info(f"Scanner: generated {len(all_signals)} raw signals, {len(filtered)} after pipeline")

        # Persist
        if self.enable_db and filtered:
            await self._save_signals(filtered)

        return filtered

    async def _scan_ticker(self, ticker: str) -> list[Signal]:
        """Scan a single ticker with all strategies."""
        # 1. Get data — try cache first, then provider
        df = await self._get_data(ticker)

        if df.empty or len(df) < 20:
            logger.warning(f"Scanner: insufficient data for {ticker}")
            return []

        # 2. Compute indicators needed by all strategies
        df = self._compute_required_indicators(df)

        # 3. Set ticker name for Signal creation
        df.index.name = ticker

        # 4. Run each strategy
        signals: list[Signal] = []
        for strategy in self.strategies:
            try:
                sigs = strategy.analyze(df)
                for s in sigs:
                    s.ticker = ticker
                signals.extend(sigs)
            except Exception as e:
                logger.debug(f"Strategy {strategy.name} error on {ticker}: {e}")
                continue

        return signals

    async def _get_data(self, ticker: str) -> pd.DataFrame:
        """Get OHLCV data, trying cache first."""
        factory = get_session_factory()
        async with factory() as session:
            df = await get_cached_bars(session, ticker, interval="1d")

        if df.empty or len(df) < 50:
            # Fetch 200 days of data
            df = await self.provider.fetch_historical(ticker, start="2024-01-01", interval="1d")
            if not df.empty and self.enable_db:
                async with factory() as session:
                    await cache_bars(session, ticker, df, interval="1d")

        return df

    def _compute_required_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute indicators required by all active strategies.

        Instead of using the registry to run named indicators, we compute
        common indicator columns directly via TA-Lib since strategies
        expect them already present on the DataFrame.
        """
        from src.indicators.macd import compute_macd
        from src.indicators.rsi import compute_rsi
        from src.indicators.moving_averages import compute_ma_cross
        from src.indicators.bollinger import compute_bollinger
        from src.indicators.kdj import compute_kdj
        from src.indicators.adx import compute_adx
        from src.indicators.obv import compute_obv
        from src.indicators.candlestick import compute_candlestick

        # Collect what's needed
        needed = set()
        for strat in self.strategies:
            for ind in strat.required_indicators:
                needed.add(ind)

        # Compute all common indicators
        try:
            if "macd" in needed:
                df = compute_macd(df)
        except Exception:
            pass
        try:
            if "rsi" in needed:
                df = compute_rsi(df)
        except Exception:
            pass
        try:
            if "ma_cross" in needed:
                df = compute_ma_cross(df)
        except Exception:
            pass
        try:
            if "bollinger" in needed:
                df = compute_bollinger(df)
        except Exception:
            pass
        try:
            if "kdj" in needed or "kdj_signal" in needed:
                df = compute_kdj(df)
        except Exception:
            pass
        try:
            if "adx" in needed or "adx_trend" in needed:
                df = compute_adx(df)
        except Exception:
            pass
        try:
            if "obv" in needed:
                df = compute_obv(df)
        except Exception:
            pass
        try:
            if "candlestick" in needed:
                df = compute_candlestick(df)
        except Exception:
            pass

        return df

    async def _save_signals(self, signals: list[Signal]) -> None:
        """Persist signals to the database."""
        factory = get_session_factory()
        async with factory() as session:
            repo = SignalRepository(session)
            await repo.save_signals(signals)
            logger.debug(f"Scanner: saved {len(signals)} signals to DB")
