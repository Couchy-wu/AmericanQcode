"""REST API for the stock screener."""

import math

import pandas as pd
from fastapi import APIRouter, Query

from src.core.config import load_watchlists
from src.data.yahoo_provider import YahooFinanceProvider
from src.indicators.macd import compute_macd
from src.indicators.rsi import compute_rsi
from src.indicators.adx import compute_adx
from src.indicators.moving_averages import compute_ma_cross

router = APIRouter()


@router.get("/screener")
async def screener(
    watchlist: str = Query("default"),
    min_rsi: float | None = Query(None),
    max_rsi: float | None = Query(None),
    strategy: str | None = Query(None),
    direction: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
):
    """Run a screener scan on a watchlist and return results.

    Computes key indicators for each ticker and returns a summary row.
    """
    wl_cfg = load_watchlists()
    tickers = wl_cfg.get(watchlist, ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"])
    tickers = tickers[:limit]

    provider = YahooFinanceProvider()
    results = []

    for ticker in tickers:
        try:
            df = await provider.fetch_historical(ticker, start="2024-06-01", interval="1d")
            if df.empty or len(df) < 30:
                continue

            # Compute indicators
            df = compute_rsi(df)
            df = compute_macd(df)
            df = compute_adx(df)
            df = compute_ma_cross(df)

            last = df.iloc[-1]
            prev = df.iloc[-2] if len(df) >= 2 else last

            price = float(last["Close"])
            prev_price = float(prev["Close"])
            change_pct = (price / prev_price - 1) * 100 if prev_price else 0

            rsi_raw = last.get("RSI", 50)
            rsi_val = float(rsi_raw) if not (isinstance(rsi_raw, float) and math.isnan(rsi_raw)) else 50

            # Apply RSI filters
            if min_rsi is not None and rsi_val < min_rsi:
                continue
            if max_rsi is not None and rsi_val > max_rsi:
                continue

            macd_signal = "Buy" if last.get("MACD_GoldenCross", False) else \
                          ("Sell" if last.get("MACD_DeathCross", False) else
                           ("Bullish" if last.get("MACD", 0) > last.get("MACD_Signal", 0) else "Bearish"))

            results.append({
                "ticker": ticker,
                "price": round(price, 2),
                "change_pct": round(change_pct, 2),
                "rsi": round(rsi_val, 1),
                "macd_signal": macd_signal,
                "volume": int(last.get("Volume", 0)),
                "adx": round(float(last.get("ADX", 0)), 1),
                "ma_cross": "Golden" if last.get("MA_GoldenCross", False) else
                            ("Death" if last.get("MA_DeathCross", False) else "None"),
            })
        except Exception:
            continue

    return {"results": results, "count": len(results)}
