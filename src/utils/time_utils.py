"""Timestamp and market session helper utilities."""

from datetime import datetime, timedelta
from typing import Optional

import pytz

EASTERN = pytz.timezone("America/New_York")
UTC = pytz.UTC


def now_et() -> datetime:
    """Get current time in Eastern timezone."""
    return datetime.now(EASTERN)


def to_eastern(dt: datetime) -> datetime:
    """Convert a datetime to Eastern timezone."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(EASTERN)


def normalize_timestamp(dt: datetime) -> datetime:
    """Return a timezone-naive UTC datetime."""
    if dt.tzinfo is not None:
        return dt.astimezone(UTC).replace(tzinfo=None)
    return dt


def floor_to_session(dt: datetime) -> datetime:
    """Floor a datetime to the nearest market session boundary."""
    et = to_eastern(dt) if dt.tzinfo is not None else dt.replace(tzinfo=EASTERN)
    market_open = et.replace(hour=9, minute=30, second=0, microsecond=0)
    if et < market_open:
        market_open -= timedelta(days=1)
        # If the previous day is a weekend, this will be adjusted by the caller
    return market_open.astimezone(UTC)


def format_pct(value: float, decimals: int = 2) -> str:
    """Format a float as a percentage string."""
    return f"{value * 100:.{decimals}f}%"


def format_currency(value: float, decimals: int = 2) -> str:
    """Format a float as USD string."""
    return f"${value:,.{decimals}f}"
