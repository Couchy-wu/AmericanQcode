"""iTick / AllTick data provider — unlimited free calls, real-time + historical.

Register for free token at: https://itick.org or https://alltick.co
Set ITICK_API_TOKEN in your .env file.

Supports US stocks (AAPL.US), HK stocks (700.HK), China A-shares (region-based).
"""

import json
import os
import time
import urllib.parse
from datetime import date, datetime

import pandas as pd
import requests

from src.data.base import AbstractDataProvider
from src.core.models import Quote


class ITickProvider(AbstractDataProvider):
    """Data provider backed by iTick / AllTick API.

    Free tier: unlimited basic quotes, ~150ms latency REST, WebSocket support.
    30+ years of historical data including minute-level bars.

    API docs: https://blog.itick.io
    Github: https://github.com/itick-org
    """

    provider_name = "itick"

    # K-line type mapping
    KLINE_MAP = {
        "1m": 1, "5m": 2, "15m": 3, "30m": 4,
        "1h": 5, "1d": 8, "1wk": 9, "1mo": 10,
    }

    BASE_URL = "https://quote.alltick.io/quote-stock-b-api"

    def __init__(self, api_token: str | None = None):
        self.api_token = api_token or os.getenv("ITICK_API_TOKEN", "")
        if not self.api_token:
            raise ValueError("ITICK_API_TOKEN is required. Get one at https://itick.org")
        self._last_call = 0.0
        self._min_interval = 0.2  # 5 calls/sec to be safe

    def _rate_limit(self):
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()

    def _ticker_code(self, ticker: str) -> str:
        """Convert ticker to iTick format: AAPL → AAPL.US"""
        t = ticker.upper()
        if "." not in t:
            t = f"{t}.US"
        return t

    async def fetch_historical(
        self,
        ticker: str,
        start: date | str,
        end: date | str | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch historical K-line data from iTick.

        Args:
            ticker: Stock symbol (e.g., 'AAPL').
            start: Start date.
            end: End date (ignored, uses query_kline_num instead).
            interval: Bar interval — '1d', '1h', '5m', etc.

        Returns:
            DataFrame with OHLCV columns.
        """
        kline_type = self.KLINE_MAP.get(interval, 8)

        # iTick returns fixed number of bars, not date range
        # For daily, 200 bars ≈ 1 year of trading days
        num_bars = {
            "1m": 390, "5m": 390, "15m": 390, "30m": 390,
            "1h": 200, "1d": 250, "1wk": 100, "1mo": 60,
        }.get(interval, 250)

        code = self._ticker_code(ticker)
        query = {
            "trace": "american_qcode",
            "data": {
                "code": code,
                "kline_type": kline_type,
                "kline_timestamp_end": 0,  # 0 = latest
                "query_kline_num": num_bars,
                "adjust_type": 0,
            },
        }

        encoded = urllib.parse.quote(json.dumps(query))
        url = f"{self.BASE_URL}/kline?token={self.api_token}&query={encoded}"

        self._rate_limit()
        try:
            resp = requests.get(
                url,
                headers={"Content-Type": "application/json"},
                timeout=15,
            )
            data = resp.json()
        except Exception:
            return pd.DataFrame()

        if data.get("code") != 200 or "data" not in data:
            return pd.DataFrame()

        klines = data["data"].get("kline_list", [])
        if not klines:
            return pd.DataFrame()

        # Parse K-lines: [timestamp, open, high, low, close, volume, ...]
        rows = []
        for k in klines:
            if len(k) >= 6:
                # iTick timestamp is in milliseconds
                ts = pd.Timestamp(k[0], unit="ms") if k[0] > 1e12 else pd.Timestamp(k[0], unit="s")
                rows.append({
                    "timestamp": ts,
                    "Open": float(k[1]),
                    "High": float(k[2]),
                    "Low": float(k[3]),
                    "Close": float(k[4]),
                    "Volume": float(k[5]) if k[5] else 0,
                })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = df.set_index("timestamp").sort_index()

        # Filter by date range if provided
        if isinstance(start, date):
            start = datetime.combine(start, datetime.min.time())
        if isinstance(end, date):
            end = datetime.combine(end, datetime.min.time())

        if start:
            df = df[df.index >= pd.Timestamp(start)]
        if end:
            df = df[df.index <= pd.Timestamp(end)]

        return df

    async def fetch_snapshot(self, tickers: list[str]) -> dict[str, Quote]:
        """Fetch real-time quotes for multiple tickers.

        iTick batch quote works by querying each ticker individually.
        """
        results: dict[str, Quote] = {}
        for ticker in tickers:
            code = self._ticker_code(ticker)
            query = {
                "trace": "american_qcode",
                "data": {"code": code},
            }
            encoded = urllib.parse.quote(json.dumps(query))
            url = f"{self.BASE_URL}/trade-tick?token={self.api_token}&query={encoded}"

            self._rate_limit()
            try:
                resp = requests.get(
                    url,
                    headers={"Content-Type": "application/json"},
                    timeout=10,
                )
                data = resp.json()

                if data.get("code") == 200 and "data" in data:
                    tick_data = data["data"]
                    price = float(tick_data.get("price", 0) or tick_data.get("last", 0))
                    prev_close = float(tick_data.get("pre_close", price) or price)
                    change = price - prev_close if prev_close else 0
                    change_pct = (change / prev_close * 100) if prev_close else 0

                    results[ticker.upper()] = Quote(
                        ticker=ticker.upper(),
                        timestamp=datetime.now(),
                        price=price,
                        change=change,
                        change_pct=change_pct,
                        volume=float(tick_data.get("volume", 0) or 0),
                    )
            except Exception:
                continue

        return results

    async def fetch_intraday(
        self,
        ticker: str,
        interval: str = "5m",
    ) -> pd.DataFrame:
        """Fetch intraday K-line data."""
        return await self.fetch_historical(ticker, date.today(), None, interval)

    async def validate_ticker(self, ticker: str) -> bool:
        """Check if a ticker exists on iTick."""
        code = self._ticker_code(ticker)
        query = {
            "trace": "validate",
            "data": {
                "code": code,
                "kline_type": 8,
                "kline_timestamp_end": 0,
                "query_kline_num": 1,
                "adjust_type": 0,
            },
        }
        encoded = urllib.parse.quote(json.dumps(query))
        url = f"{self.BASE_URL}/kline?token={self.api_token}&query={encoded}"

        self._rate_limit()
        try:
            resp = requests.get(url, headers={"Content-Type": "application/json"}, timeout=10)
            data = resp.json()
            return data.get("code") == 200 and len(data.get("data", {}).get("kline_list", [])) > 0
        except Exception:
            return False
