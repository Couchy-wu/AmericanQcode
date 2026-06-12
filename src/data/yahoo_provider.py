"""Yahoo Finance data provider using the yfinance library."""

from datetime import date, datetime

import pandas as pd
import yfinance as yf

from src.data.base import AbstractDataProvider
from src.core.models import Quote


class YahooFinanceProvider(AbstractDataProvider):
    """Data provider backed by Yahoo Finance (yfinance).

    Pros: No API key required, unlimited historical data, good for development.
    Cons: 15-min delayed quotes, no official SLA, rate limits are informal.
    """

    provider_name = "yahoo"

    def __init__(self):
        self._ticker_cache: dict[str, yf.Ticker] = {}

    def _get_ticker(self, symbol: str) -> yf.Ticker:
        symbol = symbol.upper()
        if symbol not in self._ticker_cache:
            self._ticker_cache[symbol] = yf.Ticker(symbol)
        return self._ticker_cache[symbol]

    async def fetch_historical(
        self,
        ticker: str,
        start: date | str,
        end: date | str | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data from Yahoo Finance."""
        if end is None:
            end = date.today()
        if isinstance(start, date):
            start = start.isoformat()
        if isinstance(end, date):
            end = end.isoformat()

        yt = self._get_ticker(ticker)
        df = yt.history(start=start, end=end, interval=interval)

        if df.empty:
            return df

        # Normalize column names
        df = df.rename(columns={
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Volume": "Volume",
        })

        # Keep required columns
        cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        return df[cols]

    async def fetch_snapshot(self, tickers: list[str]) -> dict[str, Quote]:
        """Fetch snapshot quotes for multiple tickers.

        Uses yfinance's Ticker.info for single quotes (batched manually).
        """
        results: dict[str, Quote] = {}
        for symbol in tickers:
            try:
                yt = self._get_ticker(symbol)
                info = yt.info
                price = info.get("regularMarketPrice") or info.get("currentPrice", 0)
                previous_close = info.get("regularMarketPreviousClose", price)
                change = price - previous_close if previous_close else 0
                change_pct = (change / previous_close * 100) if previous_close else 0

                results[symbol.upper()] = Quote(
                    ticker=symbol.upper(),
                    timestamp=datetime.now(),
                    price=price,
                    change=change,
                    change_pct=change_pct,
                    volume=info.get("regularMarketVolume", 0),
                    bid=info.get("bid"),
                    ask=info.get("ask"),
                )
            except Exception:
                continue

        return results

    async def fetch_intraday(
        self,
        ticker: str,
        interval: str = "5m",
    ) -> pd.DataFrame:
        """Fetch recent intraday bars (1 day of data)."""
        yt = self._get_ticker(ticker)
        # Valid intervals for yfinance: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h
        # 1m max 7 days, others max 60 days
        period_map = {
            "1m": "7d",
            "2m": "60d",
            "5m": "60d",
            "15m": "60d",
            "30m": "60d",
            "1h": "60d",
        }
        period = period_map.get(interval, "5d")
        df = yt.history(period=period, interval=interval)

        if df.empty:
            return df

        cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        return df[cols]

    async def validate_ticker(self, ticker: str) -> bool:
        """Check if a ticker exists in Yahoo Finance."""
        try:
            yt = self._get_ticker(ticker)
            info = yt.info
            return info.get("symbol") is not None and info.get("regularMarketPrice") is not None
        except Exception:
            return False
