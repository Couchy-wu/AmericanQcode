"""Walk-forward backtesting engine with slippage, commission, and performance metrics."""

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from src.core.models import BacktestResult, Trade, OrderSide, SignalDirection

# Risk-free rate for Sharpe ratio (approximate US T-bill)
_RISK_FREE_RATE = 0.04


class Backtester:
    """Walk-forward backtester for a single ticker and strategy.

    Simulates trading based on a strategy's signals, computing
    performance metrics including Sharpe ratio, max drawdown, etc.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission_per_share: float = 0.005,
        slippage_bps: float = 5.0,
        position_size_pct: float = 0.2,
        benchmark_ticker: Optional[str] = "SPY",
    ):
        self.initial_capital = initial_capital
        self.commission_per_share = commission_per_share
        self.slippage_bps = slippage_bps  # basis points
        self.position_size_pct = position_size_pct
        self.benchmark_ticker = benchmark_ticker

    def run(
        self,
        df: pd.DataFrame,
        strategy_name: str,
        signals: list[dict],
        ticker: str = "UNKNOWN",
    ) -> BacktestResult:
        """Run a backtest on OHLCV data with pre-generated signals.

        Args:
            df: OHLCV DataFrame with DateTimeIndex and Close, etc.
            strategy_name: Name of the strategy.
            signals: List of dicts with keys: timestamp, direction, price.
            ticker: Stock symbol.

        Returns:
            BacktestResult with all performance metrics.
        """
        if df.empty or not signals:
            return self._empty_result(strategy_name, ticker, df)

        # Convert signals to DataFrame aligned by timestamp
        sig_df = pd.DataFrame(signals)
        sig_df["timestamp"] = pd.to_datetime(sig_df["timestamp"])
        sig_df = sig_df.set_index("timestamp").sort_index()

        # Simulate trading
        trades: list[Trade] = []
        cash = self.initial_capital
        position = 0.0  # Current shares held
        equity_curve: list[float] = []
        entry_price = 0.0

        for i, (ts, bar) in enumerate(df.iterrows()):
            price = float(bar["Close"])

            # Check for signals on this bar
            if ts in sig_df.index:
                sig_row = sig_df.loc[ts]
                # Handle multiple signals on same bar
                if isinstance(sig_row, pd.DataFrame):
                    sig_row = sig_row.iloc[-1]  # Take the last signal

                direction = sig_row.get("direction", "BULLISH")

                # Close existing position if opposite signal
                if position > 0 and direction in ("BEARISH", SignalDirection.BEARISH):
                    exit_price = self._apply_slippage(price, is_entry=False)
                    pnl = (exit_price - entry_price) * position - self.commission_per_share * position
                    pnl_pct = (exit_price / entry_price - 1) * 100 if entry_price > 0 else 0
                    cash += exit_price * position
                    trades.append(Trade(
                        entry_time=ts, exit_time=ts, ticker=ticker,
                        side=OrderSide.SELL, entry_price=entry_price,
                        exit_price=exit_price, quantity=position,
                        pnl=pnl, pnl_pct=pnl_pct, holding_bars=0,
                    ))
                    position = 0.0

                # Open new position if signal direction matches
                if position == 0 and direction in ("BULLISH", SignalDirection.BULLISH):
                    entry_price = self._apply_slippage(price, is_entry=True)
                    position_size_dollars = cash * self.position_size_pct
                    position = position_size_dollars / entry_price
                    cash -= entry_price * position + self.commission_per_share * position
                    trades.append(Trade(
                        entry_time=ts, exit_time=ts, ticker=ticker,
                        side=OrderSide.BUY, entry_price=entry_price,
                        exit_price=entry_price, quantity=position,
                        pnl=0, pnl_pct=0, holding_bars=0,
                    ))

            # Track equity
            equity = cash + position * price
            equity_curve.append(equity)

        # Close any remaining position at last price
        if position > 0 and len(df) > 0:
            last_price = float(df["Close"].iloc[-1])
            pnl = (last_price - entry_price) * position - self.commission_per_share * position
            cash += last_price * position
            position = 0.0

        # Final equity
        final_equity = cash
        total_return = (final_equity / self.initial_capital) - 1.0

        # Metrics from equity curve
        equity_series = pd.Series(equity_curve)
        returns = equity_series.pct_change().dropna()

        sharpe = self._calc_sharpe(returns)
        max_dd = self._calc_max_drawdown(equity_series)
        win_rate, profit_factor, avg_win, avg_loss = self._calc_trade_metrics(trades)
        cagr = self._calc_cagr(equity_series, df)

        best_trade = max([t.pnl for t in trades if t.pnl != 0], default=0)
        worst_trade = min([t.pnl for t in trades if t.pnl != 0], default=0)

        return BacktestResult(
            strategy=strategy_name,
            ticker=ticker,
            start_date=df.index[0],
            end_date=df.index[-1],
            initial_capital=self.initial_capital,
            final_capital=final_equity,
            total_return=total_return,
            total_return_pct=total_return * 100,
            cagr=cagr,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd * 100,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len([t for t in trades if t.pnl != 0]),
            winning_trades=len([t for t in trades if t.pnl > 0]),
            losing_trades=len([t for t in trades if t.pnl < 0]),
            avg_win=avg_win,
            avg_loss=avg_loss,
            best_trade=best_trade,
            worst_trade=worst_trade,
        )

    def _apply_slippage(self, price: float, is_entry: bool = True) -> float:
        """Apply slippage: entry fills slightly higher, exit fills slightly lower."""
        slip = price * self.slippage_bps / 10000.0
        if is_entry:
            return price + slip
        return price - slip

    def _calc_sharpe(self, returns: pd.Series) -> float:
        """Calculate annualized Sharpe ratio."""
        if returns.empty or returns.std() == 0:
            return 0.0
        excess = returns.mean() * 252 - _RISK_FREE_RATE
        vol = returns.std() * np.sqrt(252)
        return excess / vol if vol > 0 else 0.0

    def _calc_max_drawdown(self, equity: pd.Series) -> float:
        """Calculate maximum drawdown as a negative fraction."""
        peak = equity.expanding().max()
        dd = (equity - peak) / peak
        return float(dd.min()) if dd.min() < 0 else 0.0

    def _calc_cagr(self, equity: pd.Series, df: pd.DataFrame) -> float:
        """Calculate Compound Annual Growth Rate."""
        years = (df.index[-1] - df.index[0]).days / 365.25
        if years <= 0:
            return 0.0
        total_return = equity.iloc[-1] / equity.iloc[0]
        return total_return ** (1 / years) - 1

    def _calc_trade_metrics(
        self, trades: list[Trade],
    ) -> tuple[float, float, float, float]:
        """Calculate win rate, profit factor, avg win/loss."""
        closed = [t for t in trades if t.pnl != 0]
        if not closed:
            return 0.0, 0.0, 0.0, 0.0

        wins = [t for t in closed if t.pnl > 0]
        losses = [t for t in closed if t.pnl < 0]

        win_rate = len(wins) / len(closed) if closed else 0.0

        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        avg_win = gross_profit / len(wins) if wins else 0.0
        avg_loss = gross_loss / len(losses) if losses else 0.0

        return win_rate, profit_factor, avg_win, avg_loss

    def _empty_result(self, strategy: str, ticker: str, df: pd.DataFrame) -> BacktestResult:
        """Return an empty/zero result."""
        return BacktestResult(
            strategy=strategy,
            ticker=ticker,
            start_date=df.index[0] if not df.empty else datetime.now(),
            end_date=df.index[-1] if not df.empty else datetime.now(),
            initial_capital=self.initial_capital,
            final_capital=self.initial_capital,
            total_return=0.0,
            total_return_pct=0.0,
            cagr=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            max_drawdown_pct=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            avg_win=0.0,
            avg_loss=0.0,
            best_trade=0.0,
            worst_trade=0.0,
        )
