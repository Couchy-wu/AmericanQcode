"""Abstract base class for all data providers."""

from abc import ABC, abstractmethod
from datetime import date, datetime

import pandas as pd

from src.core.models import OHLCVBar, Quote


class AbstractDataProvider(ABC):
    """Contract for data providers.

    All providers must implement fetch_historical, fetch_snapshot, and fetch_intraday.
    """

    provider_name: str = "base"

    @abstractmethod
    async def fetch_historical(
        self,
        ticker: str,
        start: date | str,
        end: date | str | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch historical OHLCV bars.

        Args:
            ticker: Stock symbol (e.g., 'AAPL').
            start: Start date.
            end: End date. None means today.
            interval: Bar interval — '1d', '1wk', '1mo', '1h', etc.

        Returns:
            DataFrame with columns: open, high, low, close, volume.
        """
        ...

    @abstractmethod
    async def fetch_snapshot(self, tickers: list[str]) -> dict[str, Quote]:
        """Fetch real-time snapshot quotes for multiple tickers.

        Args:
            tickers: List of stock symbols.

        Returns:
            Dict mapping ticker symbol to Quote object.
        """
        ...

    @abstractmethod
    async def fetch_intraday(
        self,
        ticker: str,
        interval: str = "5m",
    ) -> pd.DataFrame:
        """Fetch recent intraday bars.

        Args:
            ticker: Stock symbol.
            interval: Bar interval — '1m', '5m', '15m', '30m', '1h'.

        Returns:
            DataFrame with columns: open, high, low, close, volume.
        """
        ...

    async def validate_ticker(self, ticker: str) -> bool:
        """Check if a ticker is valid/available from this provider."""
        try:
            df = await self.fetch_historical(ticker, date.today(), interval="1d")
            return not df.empty
        except Exception:
            return False
