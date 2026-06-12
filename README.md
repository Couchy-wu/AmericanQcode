# 📈 AmericanQcode

美股量化交易策略系统 — 实时数据获取、多指标技术分析、策略信号生成、CLI + Web Dashboard。

## ✨ 功能

- **多数据源**：Finnhub / iTick / Finviz / Yahoo Finance / Polygon.io，自动降级切换
- **10 种技术指标**：MACD、RSI、KDJ、布林带、ADX、OBV、VWAP、均线系统、K线形态识别（61种）、支撑阻力位
- **6 个交易策略**：MACD 金叉、RSI 背离、MA 突破、布林挤压、K线形态、复合投票
- **回测引擎**：Walk-forward 回测，含滑点/佣金，输出 Sharpe、MaxDD、CAGR、胜率
- **CLI 命令行**：`qt scan` / `qt backtest` / `qt live` / `qt report`
- **Web Dashboard**：FastAPI + Plotly.js K 线图 + WebSocket 实时信号推送
- **$10,000 持仓推荐**：多因子加权评分 + 仓位分配

## 🚀 快速开始

### 1. 安装

```bash
git clone https://github.com/Couchy-wu/AmericanQcode.git
cd AmericanQcode
pip install -r requirements.txt
```

> macOS 需要 `brew install ta-lib` 才能安装 TA-Lib。如不需要完整指标库，核心回测和推荐脚本无需 TA-Lib。

### 2. 配置数据源（可选）

```bash
cp .env.example .env
# 编辑 .env，填入免费 API Key：
#   Finnhub: https://finnhub.io/register      (60次/分钟)
#   iTick:   https://itick.org                (无限调用)
```

不配置 API Key 也能用 — 系统自动使用 Finviz 快照数据。

### 3. 运行

```bash
# 实时持仓推荐（无需 API Key）
python recommend_live.py

# 策略回测（$10,000，过去1年，10只美股，8个策略）
python backtest_runner.py

# CLI 回测单只股票
qt backtest --ticker NVDA --strategy macd_cross --capital 10000

# 启动 Web Dashboard
make run-web
# 打开 http://localhost:8080
```

## 📋 CLI 命令

```bash
# 扫描股票池，输出交易信号
qt scan --watchlist tech --strategy macd_cross --top 10

# 回测策略
qt backtest --ticker AAPL --strategy macd_cross --from 2025-01-01 --capital 10000

# 实时监控（定时扫描 + Web Dashboard）
qt live --watchlist default --interval 5 --web-port 8080

# 信号历史查询
qt report --type signals --ticker AAPL --format table
```

## 📊 Web Dashboard

| 页面 | 路径 | 功能 |
|------|------|------|
| Dashboard | `/` | K线图 + 实时信号推送（WebSocket） |
| Chart | `/chart/AAPL` | 全屏K线图，含 MACD/RSI/布林带子图 |
| Screener | `/screener` | 股票筛选器，可排序筛选 |
| Signals | `/signals` | 信号历史记录 |

## 🏗️ 架构

```
数据层 (src/data/)
  ├── FinnhubProvider / ITickProvider / YahooFinanceProvider
  └── Cache (SQLite) + Repository

指标引擎 (src/indicators/)
  ├── MACD / RSI / KDJ / Bollinger / ADX / OBV / VWAP
  ├── Candlestick (61 patterns via TA-Lib)
  └── Support/Resistance detection

策略引擎 (src/strategies/)
  ├── MACD Cross / RSI Divergence / MA Breakout
  ├── Bollinger Squeeze / Candlestick Pattern
  └── Composite (AND/OR/Weighted voting)

回测 & 扫描 (src/engine/)
  ├── Scanner: provider → indicators → strategies → signals
  ├── Backtester: walk-forward + slippage + commission
  ├── Signal Pipeline: dedup → filter → rank
  └── Scheduler: APScheduler periodic scanning

CLI (src/cli/)              Web (src/web/)
  qt scan                      FastAPI + Jinja2 + Plotly.js
  qt backtest                  REST API + WebSocket
  qt live / qt report
```

## 📊 回测策略优化对比

基于 $10,000 本金、10 只美股、过去 1 年的回测结果：

| 策略 | 原始收益 | 优化后收益 | 优化胜率 | 交易数变化 |
|------|---------|-----------|---------|-----------|
| KDJ | +0.56% | **+5.31%** ⬆️10x | 39.6% → **51.7%** | 152 → **20** (-87%) |
| BB+MACD | -3.06% | **+6.20%** ⬆️ | 38.7% → 33.3% | 36 → **14** (-61%) |
| RSI | +1.10% | +0.35% | 72.2% → **61.9%** | 12 → **32** (+167%) |

优化核心：ADX 趋势过滤砍掉 87% 的 KDJ 噪音信号，收益提升 10 倍。

## 🔌 数据源

| 数据源 | 免费额度 | K线数据 | 实时性 | 需要 Key |
|--------|---------|---------|--------|----------|
| **Finnhub** | 60次/分钟 | ✅ | 20分延迟 | ✅ 免费注册 |
| **iTick** | 无限 | ✅ | 实时 | ✅ 免费注册 |
| **Alpha Vantage** | 5次/分钟 | ✅ 20+年 | 15分延迟 | ✅ 免费注册 |
| **Finviz** | 无限 | ❌ 仅快照 | 15分延迟 | ❌ |
| **Yahoo Finance** | 非官方 | ✅ | 延迟 | ❌ |
| **Polygon.io** | 5次/分钟 | ✅ | 实时 | ✅ |

## 📁 项目结构

```
AmericanQcode/
├── config/              # YAML 配置（策略参数、股票池、全局设置）
├── src/
│   ├── core/            # 配置加载、数据模型、交易日历
│   ├── data/            # 数据获取（5个 Provider） + 缓存 + 数据库
│   ├── indicators/      # 10 个技术指标（纯函数，无副作用）
│   ├── strategies/      # 6 个交易策略（Strategy ABC）
│   ├── engine/          # 扫描器、回测器、信号管道、调度器
│   ├── cli/             # Click 命令行
│   ├── web/             # FastAPI + Jinja2 + Plotly.js
│   └── utils/           # 日志、并行、时间工具
├── tests/               # pytest 测试
├── backtest_runner.py   # 独立回测脚本（无需 TA-Lib）
├── recommend_live.py    # 实时持仓推荐
└── recommend.py         # 策略信号扫描
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request。

## ⚠️ 免责声明

本系统仅供学习和研究使用。所有交易信号均为算法生成，**不构成投资建议**。回测收益不代表未来表现。股市有风险，投资需谨慎。

---

**License**: MIT
