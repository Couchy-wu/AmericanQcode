"""APScheduler-based job scheduler for periodic scanning and maintenance."""

import asyncio
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger

from src.core.market_calendar import is_market_open, get_market_status
from src.data.database import get_session_factory
from src.data.cache import clear_stale_cache
from src.engine.scanner import Scanner


class MarketScheduler:
    """Manages periodic scanning jobs for the quant system.

    Supports:
        - Interval-based scanning (every N minutes).
        - Market-hours-only scanning.
        - Daily maintenance jobs (cache cleanup, signal expiration).
    """

    def __init__(self, scanner: Scanner):
        self.scanner = scanner
        self.scheduler = AsyncIOScheduler()
        self._scan_job_id: str | None = None
        self._signal_callbacks: list = []

    def on_signals(self, callback):
        """Register a callback to be invoked when new signals are generated.

        Callback signature: async def callback(signals: list[Signal]) -> None
        """
        self._signal_callbacks.append(callback)

    async def _scan_job(self):
        """The main scan job executed by the scheduler."""
        if not is_market_open():
            logger.debug("Scheduler: market closed, skipping scan")
            return

        logger.info("Scheduler: starting scan cycle")
        try:
            signals = await self.scanner.scan()

            # Notify callbacks (e.g., WebSocket broadcast)
            if signals and self._signal_callbacks:
                for cb in self._signal_callbacks:
                    try:
                        await cb(signals)
                    except Exception as e:
                        logger.error(f"Scheduler: callback error: {e}")

            logger.info(f"Scheduler: scan complete, {len(signals)} signals generated")
        except Exception as e:
            logger.error(f"Scheduler: scan failed: {e}")

    async def _cleanup_job(self):
        """Daily maintenance: clear stale intraday cache."""
        logger.info("Scheduler: running daily cleanup")
        try:
            factory = get_session_factory()
            async with factory() as session:
                count = await clear_stale_cache(session, max_age_days=2, interval="5m")
                logger.info(f"Scheduler: removed {count} stale cache entries")
        except Exception as e:
            logger.error(f"Scheduler: cleanup failed: {e}")

    def start(
        self,
        interval_minutes: int = 5,
        market_session_only: bool = True,
        web_port: int = 8080,
    ):
        """Start the scheduler with all configured jobs.

        Args:
            interval_minutes: Minutes between scan cycles.
            market_session_only: If True, scan only during market hours.
            web_port: Port for the web dashboard (informational).
        """
        # Main scan job
        if market_session_only:
            # During market hours: every N minutes, Mon-Fri
            self._scan_job_id = self.scheduler.add_job(
                self._scan_job,
                trigger=CronTrigger(
                    day_of_week="mon-fri",
                    hour="9-16",
                    minute=f"*/{interval_minutes}",
                ),
                id="market_scan",
            ).id
        else:
            self._scan_job_id = self.scheduler.add_job(
                self._scan_job,
                trigger=IntervalTrigger(minutes=interval_minutes),
                id="market_scan",
            ).id

        # Daily cleanup at 1:05 AM ET
        self.scheduler.add_job(
            self._cleanup_job,
            trigger=CronTrigger(hour=1, minute=5),
            id="daily_cleanup",
        )

        self.scheduler.start()
        logger.info(
            f"Scheduler: started (interval={interval_minutes}min, "
            f"market_only={market_session_only})"
        )

        # Show next market open/close
        status = get_market_status()
        logger.info(f"Market status: {status.status.value}")
        if status.next_open:
            logger.info(f"Next market open: {status.next_open}")
        if status.next_close:
            logger.info(f"Next market close: {status.next_close}")

    def stop(self):
        """Stop the scheduler."""
        self.scheduler.shutdown(wait=False)
        logger.info("Scheduler: stopped")

    async def run_once(self) -> list:
        """Run a single scan cycle manually (for testing)."""
        return await self.scanner.scan()
