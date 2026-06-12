"""CLI scan command: scan a watchlist and display generated signals."""

import asyncio
import json

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.core.config import load_watchlists, get_strategy_configs
from src.core.models import SignalDirection
from src.data.yahoo_provider import YahooFinanceProvider
from src.engine.scanner import Scanner
from src.engine.signal_pipeline import SignalPipeline
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


@click.command("scan")
@click.option("--watchlist", "-w", default="default", help="Watchlist name or comma-separated tickers")
@click.option("--strategy", "-s", default="all", help="Strategy name or 'all'")
@click.option("--output", "-o", type=click.Choice(["table", "json", "csv"]), default="table")
@click.option("--top", "-n", type=int, default=20, help="Show top N signals")
@click.option("--min-confidence", "-c", type=float, default=0.5, help="Minimum confidence threshold")
def scan_cmd(watchlist, strategy, output, top, min_confidence):
    """Scan a watchlist for trading signals."""
    console.print(Panel.fit("[bold blue]AmericanQcode Scanner[/bold blue]", border_style="blue"))

    # Load watchlist
    wl_cfg = load_watchlists()
    if watchlist in wl_cfg:
        tickers = wl_cfg[watchlist]
        console.print(f"Watchlist: [cyan]{watchlist}[/cyan] ({len(tickers)} tickers)")
    else:
        tickers = [t.strip().upper() for t in watchlist.split(",")]
        console.print(f"Tickers: [cyan]{', '.join(tickers)}[/cyan]")

    # Load strategies
    strategy_configs = get_strategy_configs()
    if strategy == "all":
        active_strategies = [name for name, cfg in strategy_configs.items() if cfg.get("enabled", True)]
    elif strategy in strategy_configs:
        active_strategies = [strategy]
    else:
        console.print(f"[red]Unknown strategy: {strategy}[/red]")
        console.print(f"Available: {', '.join(strategy_configs.keys())}")
        return

    strategies = []
    for s_name in active_strategies:
        if s_name in STRATEGY_MAP:
            cfg = strategy_configs.get(s_name, {})
            params = cfg.get("params", {})
            strategies.append(STRATEGY_MAP[s_name](**params))
        else:
            console.print(f"[yellow]Warning: no implementation for '{s_name}'[/yellow]")

    if not strategies:
        console.print("[red]No strategies to run.[/red]")
        return

    console.print(f"Strategies: [green]{', '.join(s.name for s in strategies)}[/green]")

    # Run scan
    async def _run():
        provider = YahooFinanceProvider()
        pipeline = SignalPipeline(min_confidence=min_confidence)
        scanner = Scanner(provider, strategies, tickers, pipeline=pipeline)
        return await scanner.scan()

    console.print("\nScanning... (this may take a moment)\n")
    try:
        signals = asyncio.run(_run())
    except Exception as e:
        console.print(f"[red]Scan failed: {e}[/red]")
        return

    if not signals:
        console.print("[yellow]No signals generated.[/yellow]")
        return

    # Sort by confidence
    signals = sorted(signals, key=lambda s: s.confidence, reverse=True)[:top]

    if output == "json":
        result = []
        for s in signals:
            result.append({
                "ticker": s.ticker,
                "direction": s.direction.value,
                "confidence": s.confidence,
                "strategy": s.strategy,
                "reasoning": s.reasoning,
                "price": s.price_at_signal,
            })
        click.echo(json.dumps(result, indent=2, default=str))
    elif output == "csv":
        click.echo("ticker,direction,confidence,strategy,price,reasoning")
        for s in signals:
            click.echo(f"{s.ticker},{s.direction.value},{s.confidence:.2f},{s.strategy},{s.price_at_signal:.2f},\"{s.reasoning}\"")
    else:
        # Rich table
        table = Table(title=f"Signals ({len(signals)})", show_lines=False)
        table.add_column("Ticker", style="cyan", width=8)
        table.add_column("Dir", width=8)
        table.add_column("Conf", width=6, justify="right")
        table.add_column("Strategy", style="green", width=20)
        table.add_column("Price", width=10, justify="right")
        table.add_column("Reasoning", style="dim", max_width=60)

        for s in signals:
            dir_style = "[green]BULL[/green]" if s.direction == SignalDirection.BULLISH else "[red]BEAR[/red]"
            conf_color = "green" if s.confidence >= 0.7 else ("yellow" if s.confidence >= 0.6 else "red")
            table.add_row(
                s.ticker,
                dir_style,
                f"[{conf_color}]{s.confidence:.2f}[/{conf_color}]",
                s.strategy,
                f"${s.price_at_signal:.2f}",
                s.reasoning[:80],
            )

        console.print(table)
