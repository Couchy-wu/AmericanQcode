"""CLI report command: query historical signals and backtest results."""

import asyncio
import json
from datetime import datetime, timedelta

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.core.models import SignalDirection
from src.data.database import get_session_factory
from src.data.repository import SignalRepository, BacktestRepository

console = Console()


@click.command("report")
@click.option("--type", "-t", "report_type", type=click.Choice(["signals", "backtest"]), default="signals")
@click.option("--ticker", "-s", default=None, help="Filter by ticker")
@click.option("--strategy", "-g", default=None, help="Filter by strategy")
@click.option("--direction", "-d", type=click.Choice(["bullish", "bearish"]), default=None)
@click.option("--from", "from_date", default=None, help="From date (YYYY-MM-DD)")
@click.option("--to", "to_date", default=None, help="To date (YYYY-MM-DD)")
@click.option("--limit", "-n", type=int, default=50, help="Max results")
@click.option("--format", "-f", "fmt", type=click.Choice(["table", "json", "csv"]), default="table")
def report_cmd(report_type, ticker, strategy, direction, from_date, to_date, limit, fmt):
    """Query historical signals or backtest results."""
    if report_type == "signals":
        _show_signals(ticker, strategy, direction, from_date, to_date, limit, fmt)
    elif report_type == "backtest":
        _show_backtests(limit, fmt)


def _show_signals(ticker, strategy, direction, from_date, to_date, limit, fmt):
    """Query and display signal history."""
    dir_enum = None
    if direction == "bullish":
        dir_enum = SignalDirection.BULLISH
    elif direction == "bearish":
        dir_enum = SignalDirection.BEARISH

    start = datetime.fromisoformat(from_date) if from_date else None
    end = datetime.fromisoformat(to_date) if to_date else None

    async def _query():
        factory = get_session_factory()
        async with factory() as session:
            repo = SignalRepository(session)
            return await repo.get_signals(
                ticker=ticker, strategy=strategy, direction=dir_enum,
                start=start, end=end, limit=limit,
            )

    try:
        signals = asyncio.run(_query())
    except Exception as e:
        console.print(f"[red]Error querying signals: {e}[/red]")
        return

    if not signals:
        console.print("[yellow]No signals found.[/yellow]")
        return

    if fmt == "json":
        result = [{
            "ticker": s.ticker,
            "timestamp": s.timestamp.isoformat(),
            "direction": s.direction.value,
            "confidence": s.confidence,
            "strategy": s.strategy,
            "reasoning": s.reasoning,
            "price": s.price_at_signal,
        } for s in signals]
        click.echo(json.dumps(result, indent=2))
    elif fmt == "csv":
        click.echo("ticker,timestamp,direction,confidence,strategy,price,reasoning")
        for s in signals:
            click.echo(f"{s.ticker},{s.timestamp},{s.direction.value},{s.confidence:.2f},{s.strategy},{s.price_at_signal:.2f},\"{s.reasoning}\"")
    else:
        table = Table(title=f"Signal History ({len(signals)})")
        table.add_column("Time", style="dim")
        table.add_column("Ticker", style="cyan")
        table.add_column("Dir")
        table.add_column("Conf", justify="right")
        table.add_column("Strategy", style="green")
        table.add_column("Price", justify="right")
        table.add_column("Reasoning", style="dim", max_width=50)
        for s in signals:
            dir_style = "[green]BULL[/green]" if s.direction == SignalDirection.BULLISH else "[red]BEAR[/red]"
            table.add_row(
                s.timestamp.strftime("%Y-%m-%d %H:%M") if s.timestamp else "",
                s.ticker or "-",
                dir_style,
                f"{s.confidence:.2f}",
                s.strategy,
                f"${s.price_at_signal:.2f}" if s.price_at_signal else "-",
                s.reasoning[:60] if s.reasoning else "",
            )
        console.print(table)


def _show_backtests(limit, fmt):
    """Query and display backtest history."""
    async def _query():
        factory = get_session_factory()
        async with factory() as session:
            repo = BacktestRepository(session)
            return await repo.get_runs(limit=limit)

    try:
        runs = asyncio.run(_query())
    except Exception as e:
        console.print(f"[red]Error querying backtests: {e}[/red]")
        return

    if not runs:
        console.print("[yellow]No backtest runs found.[/yellow]")
        return

    if fmt == "json":
        click.echo(json.dumps(runs, indent=2, default=str))
    else:
        table = Table(title="Backtest History")
        table.add_column("Date", style="dim")
        table.add_column("Strategy")
        table.add_column("Ticker", style="cyan")
        table.add_column("Period")
        table.add_column("Return", justify="right")
        table.add_column("Sharpe", justify="right")
        table.add_column("MaxDD", justify="right")
        table.add_column("Win Rate", justify="right")
        for r in runs:
            metrics = r.get("metrics", {})
            ret_pct = metrics.get("total_return_pct", 0)
            ret_color = "green" if ret_pct > 0 else "red"
            table.add_row(
                str(r.get("created_at", ""))[:19],
                r.get("strategy", ""),
                r.get("ticker", ""),
                r.get("date_range", ""),
                f"[{ret_color}]{ret_pct:+.2f}%[/{ret_color}]" if isinstance(ret_pct, (int, float)) else "-",
                f"{metrics.get('sharpe_ratio', 0):.2f}",
                f"[red]{metrics.get('max_drawdown_pct', 0):.2f}%[/red]",
                f"{metrics.get('win_rate', 0) * 100:.1f}%" if metrics.get('win_rate') else "-",
            )
        console.print(table)
