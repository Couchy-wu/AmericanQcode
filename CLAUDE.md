# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AmericanQcode is a Python quantitative trading system for US stocks. It fetches real-time/historical stock data via yfinance (primary) and Polygon.io (fallback), computes technical indicators (MACD, RSI, KDJ, Bollinger Bands, ADX, OBV, VWAP, candlestick patterns, support/resistance), runs configurable trading strategies, and displays results via CLI (`qt`) and a FastAPI web dashboard with Plotly.js charts.

## Build, Lint, Test

```bash
make install-dev    # install all deps including dev tools
make test           # run pytest tests/
make test-cov       # run tests with coverage
make lint           # ruff check src/ tests/
make format         # ruff format src/ tests/
make run-web        # start web dashboard on port 8080
make run-cli-scan   # run a quick scan against default watchlist
```

To run a specific test file: `pytest tests/test_indicators/test_macd.py -v`

## CLI Commands (`qt`)

```
qt scan --watchlist tech --strategy macd_cross --top 10
qt backtest --ticker AAPL --strategy macd_cross --from 2023-01-01 --capital 100000
qt live --watchlist default --interval 5 --web-port 8080
qt report --type signals --format table --limit 50
```

## Architecture

### Data Flow

```
YahooFinanceProvider / PolygonProvider
    → fetch_historical() / fetch_snapshot() → pd.DataFrame (OHLCV)
    → Cache (SQLite via src/data/cache.py)
    → Indicator Engine (src/indicators/*.py) appends columns (RSI, MACD, etc.)
    → Strategy.analyze(df) (src/strategies/*.py) → list[Signal]
    → SignalPipeline (dedup → filter → rank)
    → CLI (Rich table) / WebSocket broadcast / SQLite persistence
```

### Key Modules

| Module | Path | Purpose |
|--------|------|---------|
| Config | `src/core/config.py` | OmegaConf YAML loader with env-var interpolation |
| Models | `src/core/models.py` | Pydantic v2: `Signal`, `OHLCVBar`, `Quote`, `BacktestResult`, `Trade` |
| Market Calendar | `src/core/market_calendar.py` | NYSE/NASDAQ trading hours via `exchange_calendars` |
| Data Provider ABC | `src/data/base.py` | `AbstractDataProvider` with `fetch_historical`, `fetch_snapshot`, `fetch_intraday` |
| Yahoo Provider | `src/data/yahoo_provider.py` | yfinance wrapper; no API key needed |
| Cache | `src/data/cache.py` | SQLite OHLCV cache with gap detection |
| Repository | `src/data/repository.py` | `SignalRepository`, `WatchlistRepository`, `BacktestRepository` |
| DB Models | `src/data/database.py` | SQLAlchemy async models: `TickerModel`, `OhlcvCacheModel`, `SignalModel`, etc. |
| Indicators | `src/indicators/` | Pure functions: compute indicator columns on a DataFrame |
| Strategies | `src/strategies/` | `Strategy` ABC with `analyze(df) → list[Signal]` |
| Scanner | `src/engine/scanner.py` | Orchestrates provider → indicators → strategies → pipeline |
| Signal Pipeline | `src/engine/signal_pipeline.py` | Dedup by (ticker, strategy), filter by min confidence, rank, cap |
| Backtester | `src/engine/backtester.py` | Walk-forward with slippage/commission; computes Sharpe, MaxDD, CAGR |
| Scheduler | `src/engine/scheduler.py` | APScheduler: periodic scanning during market hours + daily cleanup |
| Web App | `src/web/app.py` | FastAPI factory with Jinja2 templates and WebSocket support |

### Indicator Modules (all registered via `@register_indicator`)

- `src/indicators/moving_averages.py` — SMA, EMA, golden/death cross
- `src/indicators/macd.py` — MACD line, signal, histogram, cross + divergence
- `src/indicators/rsi.py` — RSI, overbought/oversold, cross + divergence
- `src/indicators/kdj.py` — KDJ (stochastic derivative), cross signals
- `src/indicators/bollinger.py` — Bollinger Bands, %B, bandwidth, squeeze detection
- `src/indicators/adx.py` — ADX, +DI, -DI, trend strength levels
- `src/indicators/obv.py` — On-Balance Volume with divergence
- `src/indicators/vwap.py` — VWAP with daily reset or rolling window
- `src/indicators/candlestick.py` — 61 TA-Lib candlestick patterns
- `src/indicators/support_resistance.py` — Swing high/low clustering for S/R levels

### Strategy Modules

- `src/strategies/macd_cross.py` — MACD golden/death cross with volume + RSI confirmation
- `src/strategies/rsi_divergence.py` — RSI oversold/overbought exit + divergence
- `src/strategies/ma_breakout.py` — MA golden/death cross with volume + ADX confirmation
- `src/strategies/bollinger_squeeze.py` — Squeeze breakout + band touch reversals
- `src/strategies/candlestick_pattern.py` — Candlestick pattern recognition signals
- `src/strategies/composite.py` — AND/OR/WEIGHTED voting across multiple strategies

### Web Routes

- `/` — Dashboard (candlestick chart + live signal feed via WebSocket)
- `/chart/{ticker}` — Full-screen chart with sub-panes (MACD, RSI, Bollinger)
- `/screener` — Stock screener table with filterable columns
- `/signals` — Signal history with pagination
- `/api/chart/{ticker}` — OHLCV + indicators JSON
- `/api/signals` — Signal query API
- `/api/screener` — Screener scan API
- `/ws/live` — WebSocket for real-time signal push

### Database (SQLite via SQLAlchemy + aiosqlite)

Tables: `tickers`, `ohlcv_cache`, `signals`, `watchlists`, `backtest_runs`, `backtest_trades`.
DB file is at `data/market.db` (gitignored). `init_db()` creates all tables on first startup.

### Config Files

- `config/settings.yaml` — Global settings (providers, polling, scanner, backtest, web, database, logging)
- `config/watchlists.yaml` — Named ticker lists (`default`, `tech`, `sp500_sample`, `test`)
- `config/strategies.yaml` — Per-strategy enable/disable flags and parameter overrides
- `.env` — API keys (`POLYGON_API_KEY`, `ALPHA_VANTAGE_KEY`)

## Development Notes

- All indicator functions take `pd.DataFrame` with OHLCV columns and return `pd.DataFrame` with added indicator columns. They are pure functions with no side effects.
- Every strategy inherits from `Strategy(ABC)` and must implement `analyze(df) -> list[Signal]`.
- The scanner computes all required indicators before running strategies — strategies assume indicator columns are already present on the DataFrame.
- The `@register_indicator` decorator adds functions to a global registry; strategies reference indicators by name.
- Use `get_session_factory()` to get a SQLAlchemy async session factory for DB operations.
- The `.vscode/settings.json` contains Claude Code environment variables for API access; do not hardcode API keys in source.
- TA-Lib requires the C library installed (`brew install ta-lib` on macOS). If unavailable, `pandas-ta` can be used as a fallback but the code expects TA-Lib function signatures.
