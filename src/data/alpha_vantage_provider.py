"""Alpha Vantage data provider — 5 calls/min free, 20+ years history, built-in indicators.

Register for free API key at: https://www.alphavantage.co/support/#api-key
Set ALPHA_VANTAGE_KEY in your .env file.

Key endpoints (free tier):
  - TIME_SERIES_DAILY / TIME_SERIES_DAILY_ADJUSTED  — daily OHLCV
  - TIME_SERIES_INTRADAY                             — intraday bars
  - GLOBAL_QUOTE                                     — real-time quote
  - Built-in indicators: MACD, RSI, BBANDS, STOCH, ADX, OBV, etc.
"""

import os
import time
from datetime import date, datetime, timedelta

import pandas as pd
import requests

from src.data.base import AbstractDataProvider
from src.core.models import Quote


class AlphaVantageProvider(AbstractDataProvider):
    """Data provider backed by Alpha Vantage.

    Free tier: 5 API calls/min, 500 calls/day.
    20+ years of historical daily data, built-in technical indicators.

    API docs: https://www.alphavantage.co/documentation/
    """

    provider_name = "alpha_vantage"

    BASE_URL = "https://www.alphavantage.co/query"

    # Interval mapping for TIME_SERIES_INTRADAY
    INTERVAL_MAP = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "60min",
    }

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_KEY", "")
        if not self.api_key:
            raise ValueError(
                "ALPHA_VANTAGE_KEY is required. "
                "Get a free key at https://www.alphavantage.co/support/#api-key"
            )
        self._last_call = 0.0
        self._min_interval = 12.0  # 5 calls/min = 1 call every 12 seconds

    def _rate_limit(self):
        """Ensure 5 calls/min limit."""
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_interval:
            wait = self._min_interval - elapsed
            time.sleep(wait)
        self._last_call = time.monotonic()

    def _get(self, params: dict) -> dict:
        """Make a rate-limited GET request to Alpha Vantage."""
        params["apikey"] = self.api_key
        self._rate_limit()
        try:
            resp = requests.get(self.BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            # Check for rate limit message
            if "Note" in data or "Information" in data:
                msg = data.get("Note") or data.get("Information", "")
                if "rate limit" in msg.lower() or "thank you for using" in msg.lower():
                    raise Exception(f"Alpha Vantage rate limit: {msg}")
            return data
        except requests.RequestException as e:
            raise Exception(f"Alpha Vantage request failed: {e}")

    async def fetch_historical(
        self,
        ticker: str,
        start: date | str,
        end: date | str | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data from Alpha Vantage.

        Daily data: TIME_SERIES_DAILY_ADJUSTED (adjusted close).
        Intraday data: TIME_SERIES_INTRADAY (extended history for recent 2 years).

        Note: Alpha Vantage's free tier returns max 100 bars per call.
        For daily data, this covers ~4 months. Future: paginate with outputsize=full.
        """
        if interval in ("1d", "1wk", "1mo"):
            return await self._fetch_daily(ticker, start, end)
        else:
            return await self._fetch_intraday_raw(ticker, interval, start, end)

    async def _fetch_daily(
        self,
        ticker: str,
        start: date | str,
        end: date | str | None = None,
    ) -> pd.DataFrame:
        """Fetch daily adjusted OHLCV."""
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": ticker.upper(),
            "outputsize": "full",  # 'compact' = 100 bars, 'full' = 20+ years
            "datatype": "json",
        }

        data = self._get(params)
        time_series_key = "Time Series (Daily)"
        if time_series_key not in data:
            # Try unadjusted
            params["function"] = "TIME_SERIES_DAILY"
            data = self._get(params)
            time_series_key = "Time Series (Daily)"

        if time_series_key not in data:
            return pd.DataFrame()

        ts_data = data[time_series_key]
        rows = []
        for date_str, values in ts_data.items():
            rows.append({
                "timestamp": pd.Timestamp(date_str),
                "Open": float(values.get("1. open", 0)),
                "High": float(values.get("2. high", 0)),
                "Low": float(values.get("3. low", 0)),
                "Close": float(values.get("5. adjusted close", values.get("4. close", 0))),
                "Volume": float(values.get("6. volume", 0)),
            })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("timestamp").sort_index()

        # Filter by date range
        if isinstance(start, date):
            start = datetime.combine(start, datetime.min.time())
        if isinstance(end, date):
            end = datetime.combine(end, datetime.min.time())

        if start:
            df = df[df.index >= pd.Timestamp(start)]
        if end:
            df = df[df.index <= pd.Timestamp(end)]

        return df

    async def _fetch_intraday_raw(
        self,
        ticker: str,
        interval: str,
        start: date | str,
        end: date | str | None = None,
    ) -> pd.DataFrame:
        """Fetch intraday bars (extended history covers 2 years)."""
        av_interval = self.INTERVAL_MAP.get(interval, "5min")

        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": ticker.upper(),
            "interval": av_interval,
            "outputsize": "full",
            "adjusted": "true",
        }

        data = self._get(params)
        time_series_key = f"Time Series ({av_interval})"
        if time_series_key not in data:
            return pd.DataFrame()

        ts_data = data[time_series_key]
        rows = []
        for date_str, values in ts_data.items():
            rows.append({
                "timestamp": pd.Timestamp(date_str),
                "Open": float(values.get("1. open", 0)),
                "High": float(values.get("2. high", 0)),
                "Low": float(values.get("3. low", 0)),
                "Close": float(values.get("4. close", 0)),
                "Volume": float(values.get("5. volume", 0)),
            })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("timestamp").sort_index()
        return df

    async def fetch_snapshot(self, tickers: list[str]) -> dict[str, Quote]:
        """Fetch real-time quotes via GLOBAL_QUOTE endpoint.

        Alpha Vantage only supports one symbol per GLOBAL_QUOTE call.
        """
        results: dict[str, Quote] = {}
        for symbol in tickers:
            try:
                params = {
                    "function": "GLOBAL_QUOTE",
                    "symbol": symbol.upper(),
                }
                data = self._get(params)

                quote_data = data.get("Global Quote", {})
                if not quote_data:
                    continue

                price = float(quote_data.get("05. price", 0))
                prev_close = float(quote_data.get("08. previous close", price))
                change = float(quote_data.get("09. change", 0))
                change_pct_str = quote_data.get("10. change percent", "0%")
                change_pct = float(change_pct_str.replace("%", ""))
                volume = int(quote_data.get("06. volume", 0))

                results[symbol.upper()] = Quote(
                    ticker=symbol.upper(),
                    timestamp=datetime.now(),
                    price=price,
                    change=change,
                    change_pct=change_pct,
                    volume=volume,
                )
            except Exception:
                continue

        return results

    async def fetch_intraday(
        self,
        ticker: str,
        interval: str = "5m",
    ) -> pd.DataFrame:
        """Fetch recent intraday bars."""
        return await self._fetch_intraday_raw(ticker, interval, date.today(), None)

    async def fetch_indicator(
        self,
        ticker: str,
        indicator: str,
        series_type: str = "close",
        **params,
    ) -> pd.DataFrame:
        """Fetch a built-in technical indicator from Alpha Vantage.

        Supported indicators (free tier):
          - MACD, MACDEXT
          - RSI, STOCH, STOCHF, STOCHRSI
          - BBANDS
          - ADX, ADXR
          - OBV
          - SMA, EMA, WMA, DEMATEMA, TEMA, TRIMA, KAMA, MAMA, T3
          - AROON, AROONOSC
          - CCI, CMO, DEMA, DX, HT_*, KAMA, MFI, MINUS_DI/DM, PLUS_DI/DM
          - MOM, NATR, PPO, ROC, SAR, ULTOSC, WILLR

        Args:
            ticker: Stock symbol.
            indicator: Indicator function name (e.g., 'MACD', 'RSI').
            series_type: Price series type ('close', 'open', 'high', 'low').
            **params: Additional parameters (time_period, series_type, fast, slow, etc.).

        Returns:
            DataFrame with indicator values.
        """
        req_params = {
            "function": indicator,
            "symbol": ticker.upper(),
            "series_type": series_type,
            **params,
        }

        data = self._get(req_params)

        # Find the indicator data key
        meta_key = "Meta Data"
        indicator_key = None
        for k in data:
            if k != meta_key and f"Technical Analysis: {indicator}" in k:
                indicator_key = k
                break
            elif k != meta_key and indicator in k:
                indicator_key = k
                break

        if not indicator_key:
            # Try any non-meta key
            for k in data:
                if k != meta_key:
                    indicator_key = k
                    break

        if not indicator_key or indicator_key not in data:
            return pd.DataFrame()

        ts_data = data[indicator_key]
        rows = []
        for date_str, values in ts_data.items():
            row = {"timestamp": pd.Timestamp(date_str)}
            for col_name, col_val in values.items():
                # Strip leading number prefix: "1. MACD" → "MACD"
                clean_name = col_name.split(". ", 1)[-1] if ". " in col_name else col_name
                row[clean_name] = float(col_val)
            rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("timestamp").sort_index()
        return df

    async def fetch_rsi(self, ticker: str, period: int = 14) -> pd.DataFrame:
        """Fetch RSI indicator from Alpha Vantage."""
        return await self.fetch_indicator(
            ticker, "RSI", series_type="close",
            time_period=period,
        )

    async def fetch_macd(
        self, ticker: str,
        fast_period: int = 12, slow_period: int = 26, signal_period: int = 9,
    ) -> pd.DataFrame:
        """Fetch MACD indicator from Alpha Vantage."""
        return await self.fetch_indicator(
            ticker, "MACD", series_type="close",
            fastperiod=fast_period, slowperiod=slow_period, signalperiod=signal_period,
        )

    async def fetch_bbands(
        self, ticker: str, period: int = 20, nbdev: float = 2.0,
    ) -> pd.DataFrame:
        """Fetch Bollinger Bands from Alpha Vantage."""
        return await self.fetch_indicator(
            ticker, "BBANDS", series_type="close",
            time_period=period, nbdevup=nbdev, nbdevdn=nbdev, matype=0,
        )

    async def validate_ticker(self, ticker: str) -> bool:
        """Check if a ticker exists on Alpha Vantage."""
        try:
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": ticker.upper(),
            }
            data = self._get(params)
            quote = data.get("Global Quote", {})
            return bool(quote and quote.get("01. symbol"))
        except Exception:
            return False
