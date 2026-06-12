"""NYSE/NASDAQ market calendar using exchange_calendars library.

Provides functions to check market status, get next open/close times,
and determine if a given datetime falls within regular trading hours.
"""

from datetime import datetime, time, timedelta
from functools import lru_cache

import pytz
from exchange_calendars import get_calendar

from src.core.models import MarketInfo, MarketStatus

# NYSE calendar covers both NYSE and NASDAQ regular trading hours
_CALENDAR_NAME = "XNYS"
_EST = pytz.timezone("America/New_York")

# Regular trading hours: 9:30 AM - 4:00 PM Eastern
_MARKET_OPEN_TIME = time(9, 30)
_MARKET_CLOSE_TIME = time(16, 0)
_PRE_MARKET_OPEN = time(4, 0)      # 4:00 AM
_AFTER_HOURS_CLOSE = time(20, 0)   # 8:00 PM


@lru_cache(maxsize=1)
def _get_calendar():
    """Get the NYSE calendar (cached)."""
    return get_calendar(_CALENDAR_NAME)


def _now_et() -> datetime:
    """Get current time in Eastern timezone."""
    return datetime.now(_EST)


def is_trading_day(date: datetime | None = None) -> bool:
    """Check if the given date (or today) is a regular trading day on NYSE."""
    cal = _get_calendar()
    dt = date or _now_et()
    session = cal.session_open(dt)
    return session is not None


def is_market_open(now: datetime | None = None) -> bool:
    """Check if the market is currently open (regular trading hours)."""
    dt = now or _now_et()
    cal = _get_calendar()
    return cal.is_open_on_minute(dt)


def get_market_status(now: datetime | None = None) -> MarketInfo:
    """Get comprehensive market status for the current time."""
    dt = now or _now_et()
    cal = _get_calendar()

    session = cal.session_open(dt)

    if session is None:
        # Holiday or weekend
        status = MarketStatus.HOLIDAY
    elif session.contains(dt):
        # Inside a session — check if in regular, pre, or after hours
        current_time = dt.time()
        open_time = session.open.time()
        close_time = session.close.time()

        if open_time <= current_time <= close_time:
            status = MarketStatus.OPEN
        elif _PRE_MARKET_OPEN <= current_time < open_time:
            status = MarketStatus.PRE_MARKET
        elif close_time < current_time <= _AFTER_HOURS_CLOSE:
            status = MarketStatus.AFTER_HOURS
        else:
            status = MarketStatus.CLOSED
    else:
        # Session exists but we're outside it
        current_time = dt.time()
        open_time = session.open.time()
        if _PRE_MARKET_OPEN <= current_time < open_time:
            status = MarketStatus.PRE_MARKET
        elif session.close.time() < current_time <= _AFTER_HOURS_CLOSE:
            status = MarketStatus.AFTER_HOURS
        else:
            status = MarketStatus.CLOSED

    # Find next open and close
    next_open: datetime | None = None
    next_close: datetime | None = None

    if status != MarketStatus.OPEN:
        # Next session open
        next_open_ts = cal.next_open(dt)
        if next_open_ts is not None:
            next_open = next_open_ts.astimezone(_EST)

    if status == MarketStatus.OPEN:
        # Current session close
        session_close = session.close.astimezone(_EST)
        next_close = session_close
    elif status in (MarketStatus.PRE_MARKET,):
        # Close of the upcoming session
        next_close_ts = cal.next_close(dt)
        if next_close_ts is not None:
            next_close = next_close_ts.astimezone(_EST)

    return MarketInfo(
        status=status,
        current_time=dt,
        next_open=next_open,
        next_close=next_close,
        timezone="America/New_York",
    )


def next_market_open(after: datetime | None = None) -> datetime | None:
    """Get the next market open time after the given datetime."""
    cal = _get_calendar()
    dt = after or _now_et()
    ts = cal.next_open(dt)
    return ts.astimezone(_EST) if ts is not None else None


def next_market_close(after: datetime | None = None) -> datetime | None:
    """Get the next market close time after the given datetime."""
    cal = _get_calendar()
    dt = after or _now_et()
    ts = cal.next_close(dt)
    return ts.astimezone(_EST) if ts is not None else None


def seconds_until_market_open() -> float:
    """Return seconds until the next market open, or 0 if already open."""
    if is_market_open():
        return 0.0
    nxt = next_market_open()
    if nxt is None:
        return float("inf")
    return (nxt - _now_et()).total_seconds()
