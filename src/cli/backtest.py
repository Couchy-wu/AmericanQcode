"""CLI backtest command: run backtests on a ticker with a strategy."""

import asyncio
from datetime import date

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.core.config import get_backtest_config
from src.data.yahoo_provider import YahooFinanceProvider
from src.engine.backtester import Backtester
from src.indicators.macd import compute_macd
from src.indicators.rsi import compute_rsi
from src.indicators.moving_averages import compute_ma_cross
from src.strategies.macd_cross import MACDCrossStrategy
from src.strategies.rsi_divergence import RSIDivergenceStrategy
from src.strategies.ma_breakout import MABreakoutStrategy

console = Console()

STRATEGY_MAP = {
    "macd_cross": (MACDCrossStrategy, ["macd"]),
    "rsi_divergence": (RSIDivergenceStrategy, ["rsi"]),
    "ma_breakout": (MABreakoutStrategy, ["ma_cross"]),
}


def _compute_indicators(df, indicator_names: list[str]):
    """Compute indicators on the DataFrame."""
    for name in indicator_names:
        try:
            if name == "macd":
                df = compute_macd(df)
            elif name == "rsi":
                df = compute_rsi(df)
            elif name == "ma_cross":
                df = compute_ma_cross(df)
            elif name == "bollinger":
                from src.indicators.bollinger import compute_bollinger
                df = compute_bollinger(df)
        except Exception:
            pass
    return df


@click.command("backtest")
@click.option("--ticker", "-t", default="SPY", help="Ticker symbol")
@click.option("--strategy", "-s", default="macd_cross", help="Strategy name")
@click.option("--from", "from_date", default="2023-01-01", help="Start date (YYYY-MM-DD)")
@click.option("--to", "to_date", default=None, help="End date (YYYY-MM-DD)")
@click.option("--capital", "-c", type=float, default=100000.0, help="Initial capital")
def backtest_cmd(ticker, strategy, from_date, to_date, capital):
    """Backtest a strategy on a ticker."""
    if strategy not in STRATEGY_MAP:
        console.print(f"[red]Unknown strategy: {strategy}[/red]")
        console.print(f"Available: {', '.join(STRATEGY_MAP.keys())}")
        return

    strategy_cls, indicators = STRATEGY_MAP[strategy]

    console.print(Panel.fit(
        f"[bold blue]Backtest: {strategy} on {ticker.upper()}[/bold blue]\n"
        f"Period: {from_date} → {to_date or 'today'} | Capital: ${capital:,.0f}",
        border_style="blue",
    ))

    bt_config = get_backtest_config()

    async def _run():
        provider = YahooFinanceProvider()
        df = await provider.fetch_historical(ticker, start=from_date, end=to_date, interval="1d")

        if df.empty:
            console.print(f"[red]No data for {ticker}[/red]")
            return None

        console.print(f"Loaded [cyan]{len(df)}[/cyan] bars")

        # Compute indicators
        df = _compute_indicators(df, indicators)
        df.index.name = ticker

        # Generate signals using strategy
        strat = strategy_cls()
        signals_raw = strat.analyze(df)
        console.print(f"Generated [yellow]{len(signals_raw)}[/yellow] raw signals")

        # Convert signals for backtest
        signal_dicts = []
        for sig in signals_raw:
            signal_dicts.append({
                "timestamp": sig.timestamp,
                "direction": sig.direction.value,
                "price": sig.price_at_signal,
            })

        # Run backtest
        backtester = Backtester(
            initial_capital=capital or bt_config.get("default_capital", 100000),
            commission_per_share=bt_config.get("commission_per_share", 0.005),
            slippage_bps=bt_config.get("slippage_bps", 5),
        )

        result = backtester.run(df, strategy, signal_dicts, ticker.upper())
        return result

    result = asyncio.run(_run())
    if result is None:
        return

    # Display results
    _print_backtest_result(result)


def _print_backtest_result(result):
    """Print backtest results as a formatted table."""
    # Performance metrics
    table = Table(title="Performance Metrics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    total_pct = result.total_return_pct
    color = "green" if total_pct > 0 else "red"

    table.add_row("Total Return", f"[{color}]{total_pct:+.2f}%[/{color}]")
    table.add_row("CAGR", f"[{color}]{result.cagr * 100:+.2f}%[/{color}]")
    table.add_row("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
    table.add_row("Max Drawdown", f"[red]{result.max_drawdown_pct:.2f}%[/red]")
    table.add_row("Win Rate", f"{result.win_rate * 100:.1f}%")
    table.add_row("Profit Factor", f"{result.profit_factor:.2f}")
    table.add_row("Total Trades", str(result.total_trades))
    table.add_row("Winning Trades", str(result.winning_trades))
    table.add_row("Losing Trades", str(result.losing_trades))
    table.add_row("Avg Win", f"${result.avg_win:,.2f}")
    table.add_row("Avg Loss", f"${result.avg_loss:,.2f}")
    table.add_row("Best Trade", f"${result.best_trade:,.2f}")
    table.add_row("Worst Trade", f"${result.worst_trade:,.2f}")

    console.print(table)

    # Summary
    final = result.final_capital
    initial = result.initial_capital
    console.print(
        f"\n[bold]Result:[/bold] ${initial:,.0f} → [{'green' if final > initial else 'red'}]${final:,.0f}[/] "
        f"({final - initial:+,.0f})"
    )
