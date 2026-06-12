"""Finnhub data provider — 60 free calls/min, real-time quotes + historical candles.

Register for free API key at: https://finnhub.io/register
Set FINNHUB_API_KEY in your .env file.
"""

import os
import time
from datetime import date, datetime, timedelta

import pandas as pd
import finnhub

from src.data.base import AbstractDataProvider
from src.core.models import Quote


class FinnhubProvider(AbstractDataProvider):
    """Data provider backed by Finnhub.io.

    Free tier: 60 API calls/minute, real-time WebSocket, 20-min delayed REST.
    Provides OHLCV candles, real-time quotes, and company fundamentals.

    API docs: https://finnhub.io/docs/api
    """

    provider_name = "finnhub"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("FINNHUB_API_KEY", "")
        if not self.api_key:
            raise ValueError("FINNHUB_API_KEY is required. Get one at https://finnhub.io/register")
        self._client = finnhub.Client(api_key=self.api_key)
        self._last_call = 0.0
        self._min_interval = 1.0  # 60 calls/min = 1 call/sec safe rate

    def _rate_limit(self):
        """Ensure we don't exceed 60 calls/min."""
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()

    async def fetch_historical(
        self,
        ticker: str,
        start: date | str,
        end: date | str | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch historical OHLCV candles from Finnhub.

        Resolution mapping:
            '1m' → '1', '5m' → '5', '15m' → '15', '30m' → '30',
            '1h' → '60', '1d' → 'D', '1wk' → 'W', '1mo' → 'M'
        """
        resolution_map = {
            "1m": "1", "5m": "5", "15m": "15", "30m": "30",
            "1h": "60", "1d": "D", "1wk": "W", "1mo": "M",
        }
        resolution = resolution_map.get(interval, "D")

        if isinstance(start, date):
            start = datetime.combine(start, datetime.min.time())
        elif isinstance(start, str):
            start = datetime.fromisoformat(start)

        if end is None:
            end = datetime.now()
        elif isinstance(end, date):
            end = datetime.combine(end, datetime.min.time())
        elif isinstance(end, str):
            end = datetime.fromisoformat(end)

        from_ts = int(start.timestamp())
        to_ts = int(end.timestamp())

        self._rate_limit()
        # Finnhub client call is synchronous, so we run it directly
        resp = self._client.stock_candles(
            ticker.upper(), resolution, from_ts, to_ts
        )

        if resp.get("s") != "ok" or not resp.get("c"):
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(resp["t"], unit="s"),
            "Open": resp["o"],
            "High": resp["h"],
            "Low": resp["l"],
            "Close": resp["c"],
            "Volume": resp["v"],
        })
        df = df.set_index("timestamp").sort_index()
        return df

    async def fetch_snapshot(self, tickers: list[str]) -> dict[str, Quote]:
        """Fetch real-time quotes for multiple tickers."""
        results: dict[str, Quote] = {}
        for symbol in tickers:
            try:
                self._rate_limit()
                resp = self._client.quote(symbol.upper())

                current = resp.get("c", 0)
                previous = resp.get("pc", current)
                change = current - previous if previous else 0
                change_pct = (change / previous * 100) if previous else 0

                results[symbol.upper()] = Quote(
                    ticker=symbol.upper(),
                    timestamp=datetime.now(),
                    price=current,
                    change=change,
                    change_pct=change_pct,
                    volume=0,  # Finnhub quote doesn't include volume
                    bid=None,
                    ask=None,
                )
            except Exception:
                continue

        return results

    async def fetch_intraday(
        self,
        ticker: str,
        interval: str = "5m",
    ) -> pd.DataFrame:
        """Fetch recent intraday bars (today's data)."""
        # Finnhub free tier has 20-min delayed intraday
        today = date.today()
        start = datetime.combine(today, datetime.min.time())
        return await self.fetch_historical(ticker, start, None, interval)

    async def fetch_quote(self, ticker: str) -> dict:
        """Get a single quote with full detail."""
        self._rate_limit()
        return self._client.quote(ticker.upper())

    async def fetch_company_profile(self, ticker: str) -> dict:
        """Get company profile (name, industry, market cap, etc.)."""
        self._rate_limit()
        return self._client.company_profile2(symbol=ticker.upper())

    async def validate_ticker(self, ticker: str) -> bool:
        """Check if a ticker is valid on Finnhub."""
        try:
            self._rate_limit()
            resp = self._client.company_profile2(symbol=ticker.upper())
            return bool(resp and resp.get("ticker"))
        except Exception:
            return False
