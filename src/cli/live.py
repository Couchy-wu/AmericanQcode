"""CLI live command: start real-time scanning with optional web dashboard."""

import asyncio
import signal
import sys

import click
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel

from src.core.config import load_watchlists, get_strategy_configs
from src.core.market_calendar import get_market_status, is_market_open
from src.core.models import SignalDirection, MarketStatus
from src.data.yahoo_provider import YahooFinanceProvider
from src.data.database import init_db
from src.engine.scanner import Scanner
from src.engine.signal_pipeline import SignalPipeline
from src.engine.scheduler import MarketScheduler
from src.strategies.macd_cross import MACDCrossStrategy
from src.strategies.rsi_divergence import RSIDivergenceStrategy
from src.strategies.ma_breakout import MABreakoutStrategy
from src.strategies.bollinger_squeeze import BollingerSqueezeStrategy
from src.strategies.candlestick_pattern import CandlestickPatternStrategy

STRATEGY_MAP = {
    "macd_cross": MACDCrossStrategy,
    "rsi_divergence": RSIDivergenceStrategy,
    "ma_breakout": MABreakoutStrategy,
    "bollinger_squeeze": BollingerSqueezeStrategy,
    "candlestick_pattern": CandlestickPatternStrategy,
}

console = Console()


@click.command("live")
@click.option("--watchlist", "-w", default="default", help="Watchlist name or comma-separated tickers")
@click.option("--interval", "-i", type=int, default=5, help="Poll interval in minutes")
@click.option("--strategies", "-s", default="all", help="Comma-separated strategy names or 'all'")
@click.option("--web-port", "-p", type=int, default=0, help="Start web dashboard on port (0 = disabled)")
@click.option("--no-web", is_flag=True, help="Disable web dashboard")
def live_cmd(watchlist, interval, strategies, web_port, no_web):
    """Start live market scanning. Press Ctrl+C to stop."""
    console.print(Panel.fit("[bold blue]AmericanQcode Live Mode[/bold blue]", border_style="blue"))

    # Load watchlist
    wl_cfg = load_watchlists()
    if watchlist in wl_cfg:
        tickers = wl_cfg[watchlist]
    else:
        tickers = [t.strip().upper() for t in watchlist.split(",")]

    console.print(f"Watchlist: [cyan]{watchlist}[/cyan] ({len(tickers)} tickers)")

    # Load strategies
    strategy_configs = get_strategy_configs()
    if strategies == "all":
        active_names = [name for name, cfg in strategy_configs.items() if cfg.get("enabled", True)]
    else:
        active_names = [s.strip() for s in strategies.split(",")]

    strategy_instances = []
    for s_name in active_names:
        if s_name in STRATEGY_MAP:
            cfg = strategy_configs.get(s_name, {})
            strategy_instances.append(STRATEGY_MAP[s_name](**cfg.get("params", {})))

    if not strategy_instances:
        console.print("[red]No strategies configured.[/red]")
        return

    console.print(f"Strategies: [green]{', '.join(s.name for s in strategy_instances)}[/green]")
    console.print(f"Poll interval: [yellow]{interval} min[/yellow]")

    # Market status
    status = get_market_status()
    _print_market_status(status)

    # Set up scanner and scheduler
    provider = YahooFinanceProvider()
    pipeline = SignalPipeline()
    scanner = Scanner(provider, strategy_instances, tickers, pipeline=pipeline, enable_db=True)
    scheduler = MarketScheduler(scanner)

    # Store latest signals for display
    latest_signals: list = []

    async def on_signals(signals):
        nonlocal latest_signals
        latest_signals = signals

    scheduler.on_signals(on_signals)

    # Handle web server
    web_task = None
    if not no_web and web_port > 0:
        async def _start_web():
            import uvicorn
            from src.web.app import create_app
            app = create_app(scanner)
            config = uvicorn.Config(app, host="0.0.0.0", port=web_port, log_level="info")
            server = uvicorn.Server(config)
            await server.serve()

        web_task = asyncio.create_task(_start_web())
        console.print(f"Web Dashboard: [cyan]http://localhost:{web_port}[/cyan]")

    # Signal handlers for graceful shutdown
    def _shutdown():
        scheduler.stop()
        if web_task:
            web_task.cancel()
        console.print("\n[yellow]Shutting down...[/yellow]")

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, lambda s, f: _shutdown())
        except Exception:
            pass

    # Start scheduler
    scheduler.start(interval_minutes=interval)

    console.print("\n[bold]Live mode active. Press Ctrl+C to stop.[/bold]\n")

    try:
        # Keep running — the scheduler handles everything
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        _shutdown()


def _print_market_status(status):
    """Print current market status."""
    color_map = {
        MarketStatus.OPEN: "green",
        MarketStatus.CLOSED: "yellow",
        MarketStatus.PRE_MARKET: "blue",
        MarketStatus.AFTER_HOURS: "magenta",
        MarketStatus.HOLIDAY: "red",
    }
    color = color_map.get(status.status, "white")
    console.print(f"Market: [bold {color}]{status.status.value}[/bold {color}]")
    if status.next_open:
        console.print(f"Next Open: {status.next_open.strftime('%Y-%m-%d %H:%M %Z')}")
    if status.next_close:
        console.print(f"Next Close: {status.next_close.strftime('%Y-%m-%d %H:%M %Z')}")
