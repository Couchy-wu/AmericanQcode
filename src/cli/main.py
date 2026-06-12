"""CLI entry point: `qt` command group using Click."""

import asyncio
import sys
from datetime import date, datetime

import click

from src.cli.scan import scan_cmd
from src.cli.backtest import backtest_cmd
from src.cli.live import live_cmd
from src.cli.report import report_cmd
from src.utils.logging_config import setup_logging


@click.group()
@click.version_option(version="0.1.0", prog_name="american-qcode")
@click.pass_context
def cli(ctx):
    """AmericanQcode — US Stock Quantitative Trading System.

    Scan for trading signals, backtest strategies, and monitor markets.
    """
    ctx.ensure_object(dict)
    setup_logging()


cli.add_command(scan_cmd, name="scan")
cli.add_command(backtest_cmd, name="backtest")
cli.add_command(live_cmd, name="live")
cli.add_command(report_cmd, name="report")


if __name__ == "__main__":
    cli()
