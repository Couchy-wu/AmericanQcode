"""REST API for chart data — OHLCV bars with computed indicators."""

from fastapi import APIRouter, Query, HTTPException

from src.data.yahoo_provider import YahooFinanceProvider
from src.indicators.macd import compute_macd
from src.indicators.rsi import compute_rsi
from src.indicators.moving_averages import compute_ma_cross
from src.indicators.bollinger import compute_bollinger
from src.indicators.kdj import compute_kdj
from src.indicators.candlestick import compute_candlestick

router = APIRouter()


@router.get("/chart/{ticker}")
async def get_chart_data(
    ticker: str,
    interval: str = Query("1d", regex="^(1d|1h|5m|15m|30m|1wk)$"),
    indicators: str = Query("ma,macd,rsi,bollinger"),
):
    """Get OHLCV data with computed indicators for a ticker.

    Args:
        ticker: Stock symbol.
        interval: Bar interval.
        indicators: Comma-separated indicator names to include.
    """
    provider = YahooFinanceProvider()

    try:
        df = await provider.fetch_historical(ticker, start="2024-01-01", interval=interval)
        if df.empty:
            # Try intraday
            df = await provider.fetch_intraday(ticker, interval=interval)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch data: {e}")

    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data for {ticker}")

    # Compute requested indicators
    indicator_list = [i.strip() for i in indicators.split(",") if i.strip()]

    for ind in indicator_list:
        try:
            if ind == "ma":
                df = compute_ma_cross(df)
            elif ind == "macd":
                df = compute_macd(df)
            elif ind == "rsi":
                df = compute_rsi(df)
            elif ind == "bollinger":
                df = compute_bollinger(df)
            elif ind == "kdj":
                df = compute_kdj(df)
            elif ind == "candlestick":
                df = compute_candlestick(df)
        except Exception:
            pass

    # Convert to JSON-friendly format
    # Keep only the last 252 bars for daily, 200 for intraday
    df = df.tail(252 if interval in ("1d", "1wk") else 200)

    result = {
        "ticker": ticker.upper(),
        "interval": interval,
        "bars": [],
        "indicators": indicator_list,
    }

    for idx, row in df.iterrows():
        bar = {
            "timestamp": str(idx),
            "open": float(row.get("Open", 0)),
            "high": float(row.get("High", 0)),
            "low": float(row.get("Low", 0)),
            "close": float(row.get("Close", 0)),
            "volume": float(row.get("Volume", 0)),
        }
        # Add indicator columns
        for col in df.columns:
            if col not in ("Open", "High", "Low", "Close", "Volume") and not col.startswith("Pattern_"):
                val = row.get(col)
                if val is not None and str(val) != 'nan':
                    bar[col] = float(val) if isinstance(val, (int, float)) else bool(val)
        result["bars"].append(bar)

    return result
