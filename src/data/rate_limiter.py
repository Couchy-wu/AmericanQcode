"""Token-bucket rate limiter for API providers."""

import asyncio
import time
from dataclasses import dataclass, field

from src.core.exceptions import RateLimitError


@dataclass
class RateLimiter:
    """Async token-bucket rate limiter.

    Tracks calls per minute and calls per day, blocking or erroring
    when limits are exceeded.
    """

    provider_name: str
    max_calls_per_minute: int = 5
    max_calls_per_day: int = 500
    wait_on_limit: bool = True
    max_wait_seconds: float = 30.0

    _minute_tokens: float = field(default=0.0, init=False)
    _day_calls: int = field(default=0, init=False)
    _last_refill: float = field(default_factory=time.monotonic, init=False)
    _day_reset: float = field(default_factory=time.monotonic, init=False)

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        refill_rate = self.max_calls_per_minute / 60.0
        self._minute_tokens = min(
            float(self.max_calls_per_minute),
            self._minute_tokens + elapsed * refill_rate,
        )
        self._last_refill = now

        # Reset daily counter after 24 hours
        if now - self._day_reset >= 86400:
            self._day_calls = 0
            self._day_reset = now

    async def acquire(self) -> None:
        """Acquire permission for one API call. Blocks or raises on limit."""
        self._refill()

        if self._day_calls >= self.max_calls_per_day:
            wait = 86400 - (time.monotonic() - self._day_reset)
            raise RateLimitError(
                self.provider_name,
                retry_after=wait,
            )

        if self._minute_tokens < 1.0:
            if self.wait_on_limit:
                wait = (1.0 - self._minute_tokens) / (self.max_calls_per_minute / 60.0)
                if wait > self.max_wait_seconds:
                    raise RateLimitError(self.provider_name, retry_after=wait)
                await asyncio.sleep(wait)
                self._refill()
            else:
                wait = 60.0
                raise RateLimitError(self.provider_name, retry_after=wait)

        self._minute_tokens -= 1.0
        self._day_calls += 1

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, *args):
        pass  # Token already consumed
