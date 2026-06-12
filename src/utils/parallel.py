"""Parallel execution utilities for multi-ticker scanning."""

import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
from typing import Callable, TypeVar

import pandas as pd

T = TypeVar("T")


async def run_parallel(
    func: Callable[..., T],
    items: list,
    max_workers: int = 10,
    use_processes: bool = False,
    *args,
    **kwargs,
) -> list[T]:
    """Run a function in parallel across items using a thread or process pool.

    Args:
        func: The function to run on each item. First arg will be the item.
        items: List of items to process.
        max_workers: Maximum concurrent workers.
        use_processes: Use ProcessPoolExecutor if True (for CPU-bound tasks),
                       else ThreadPoolExecutor (for I/O-bound tasks).
        *args, **kwargs: Additional args/kwargs passed to func after item.

    Returns:
        List of results in the same order as items. Exceptions produce None.
    """
    if not items:
        return []

    pool_cls = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    loop = asyncio.get_running_loop()

    async def _run_one(item):
        try:
            fn = partial(func, item, *args, **kwargs)
            if asyncio.iscoroutinefunction(func):
                return await fn()
            else:
                return await loop.run_in_executor(None, fn)
        except Exception:
            return None

    # Limit concurrency with a semaphore
    sem = asyncio.Semaphore(max_workers)

    async def _bounded(item):
        async with sem:
            return await _run_one(item)

    tasks = [_bounded(item) for item in items]
    return await asyncio.gather(*tasks)


async def parallel_scan_tickers(
    scanner_func: Callable,
    tickers: list[str],
    max_workers: int = 10,
    **kwargs,
) -> list:
    """Scan multiple tickers in parallel. Each ticker is processed by scanner_func."""

    async def _scan_one(ticker: str):
        try:
            return await scanner_func(ticker, **kwargs)
        except Exception:
            return None

    sem = asyncio.Semaphore(max_workers)

    async def _bounded(ticker):
        async with sem:
            return await _scan_one(ticker)

    tasks = [_bounded(t) for t in tickers]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]
