#!/usr/bin/env python3
"""
¥10,000,000 持仓推荐 — 基于 Tiingo 真实数据 + 全套技术指标 + 多策略加权评分
"""

import os, sys, time, json, subprocess, warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Load .env
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ[_k.strip()] = _v.strip()

# ─── Config ──────────────────────────────────────────────────────────────────

CAPITAL = 10_000_000      # ¥1000万
MAX_POSITIONS = 8          # 分散到8个标的
MAX_SINGLE_PCT = 0.18      # 单票不超过 18%
MIN_CONFIDENCE = 0.40

WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA",
    "AMD", "INTC", "CRM", "ADBE", "NFLX", "DIS", "PYPL", "UBER",
    "SPY", "QQQ", "IWM", "DIA",
]

# ─── Data Fetch (Tiingo) ─────────────────────────────────────────────────────


def fetch_tiingo(ticker: str) -> Optional[pd.DataFrame]:
    key = os.getenv("TIINGO_API_KEY", "")
    if not key:
        return None
    try:
        start = (datetime.now() - timedelta(days=200)).strftime("%Y-%m-%d")
        url = f"https://api.tiingo.com/tiingo/daily/{ticker.upper()}/prices?startDate={start}&token={key}"
        r = subprocess.run(["curl", "-s", "--max-time", "12", url], capture_output=True, text=True, timeout=15)
        data = json.loads(r.stdout)
        if not isinstance(data, list) or len(data) < 50:
            return None
        rows = [{"timestamp": pd.Timestamp(b["date"]), "Open": float(b["adjOpen"]),
                 "High": float(b["adjHigh"]), "Low": float(b["adjLow"]),
                 "Close": float(b["adjClose"]), "Volume": float(b["adjVolume"])} for b in data]
        return pd.DataFrame(rows).set_index("timestamp").sort_index()
    except:
        return None


# ─── Indicators ──────────────────────────────────────────────────────────────


def ema(s, p): return s.ewm(span=p, adjust=False).mean()
def sma(s, p): return s.rolling(p).mean()


def compute_indicators(df):
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    ml = ema(c, 12) - ema(c, 26)
    sl = ema(ml, 9)
    df["MACD"], df["MACD_Signal"], df["MACD_Histogram"] = ml, sl, ml - sl
    df["MACD_GoldenCross"] = (ml > sl) & (ml.shift(1) <= sl.shift(1))
    df["MACD_DeathCross"] = (ml < sl) & (ml.shift(1) >= sl.shift(1))

    d = c.diff()
    g, lo = d.where(d > 0, 0.0), (-d).where(d < 0, 0.0)
    rs = g.ewm(alpha=1/14, adjust=False).mean() / lo.ewm(alpha=1/14, adjust=False).mean()
    df["RSI"] = 100 - 100 / (1 + rs)

    df["MA_20"] = ema(c, 20)
    df["MA_50"] = ema(c, 50)
    df["MA_GoldenCross"] = (df["MA_20"] > df["MA_50"]) & (df["MA_20"].shift(1) <= df["MA_50"].shift(1))
    df["MA_DeathCross"] = (df["MA_20"] < df["MA_50"]) & (df["MA_20"].shift(1) >= df["MA_50"].shift(1))

    bb_mid = sma(c, 20)
    bb_std = c.rolling(20).std()
    df["BB_Upper"], df["BB_Middle"], df["BB_Lower"] = bb_mid + 2*bb_std, bb_mid, bb_mid - 2*bb_std

    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    um, dm = h - h.shift(1), l.shift(1) - l
    pdi = 100 * pd.Series(np.where((um > dm) & (um > 0), um, 0), index=df.index).ewm(alpha=1/14, adjust=False).mean() / atr
    mdi = 100 * pd.Series(np.where((dm > um) & (dm > 0), dm, 0), index=df.index).ewm(alpha=1/14, adjust=False).mean() / atr
    df["ADX"] = (100 * (pdi - mdi).abs() / (pdi + mdi + 1e-10)).ewm(alpha=1/14, adjust=False).mean()
    df["DI_Plus"], df["DI_Minus"] = pdi, mdi

    ll, hh = l.rolling(9).min(), h.rolling(9).max()
    rsv = (c - ll) / (hh - ll + 1e-10) * 100
    k = rsv.ewm(alpha=1/3, adjust=False).mean()
    d2 = k.ewm(alpha=1/3, adjust=False).mean()
    df["KDJ_K"], df["KDJ_D"], df["KDJ_J"] = k, d2, 3*k - 2*d2
    return df


# ─── Scoring Engine ──────────────────────────────────────────────────────────


def score_ticker(df, ticker):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    price = float(last["Close"])
    avg_vol = float(df["Volume"].rolling(20).mean().iloc[-1])
    vol_ratio = float(last["Volume"]) / avg_vol if avg_vol > 0 else 1.0

    b, be = 0.0, 0.0
    rb, rbe = [], []

    # ── MACD (25pts) ──
    if last.get("MACD_GoldenCross", False):
        b += 25; rb.append("MACD金叉⭐")
    elif last["MACD"] > last["MACD_Signal"]:
        b += 12
        if last["MACD_Histogram"] > prev.get("MACD_Histogram", 0):
            b += 5; rb.append("MACD多头增强")
    if last.get("MACD_DeathCross", False):
        be += 25; rbe.append("MACD死叉⭐")
    elif last["MACD"] < last["MACD_Signal"]:
        be += 12

    # ── RSI (20pts) ──
    rsi = last.get("RSI", 50)
    prsi = prev.get("RSI", 50)
    if rsi < 30: b += 18; rb.append(f"RSI超卖({rsi:.0f})⭐")
    elif rsi < 40 and rsi > prsi: b += 8; rb.append(f"RSI反弹({rsi:.0f})")
    elif rsi < 45: b += 4
    if rsi > 70: be += 18; rbe.append(f"RSI超买({rsi:.0f})⭐")
    elif rsi > 60 and rsi < prsi: be += 8; rbe.append(f"RSI回落({rsi:.0f})")

    # RSI divergence
    if len(df) >= 20:
        p1, p2 = df["Close"].iloc[-20:-10], df["Close"].iloc[-10:]
        r1, r2 = df["RSI"].iloc[-20:-10], df["RSI"].iloc[-10:]
        if p2.min() < p1.min() and r2.min() > r1.min():
            b += 20; rb.append("RSI看涨背离⭐⭐")
        if p2.max() > p1.max() and r2.max() < r1.max():
            be += 20; rbe.append("RSI看跌背离⭐⭐")

    # ── MA Trend (20pts) ──
    ma20, ma50 = last["MA_20"], last["MA_50"]
    if price > ma20 > ma50: b += 14; rb.append("多头排列(价>MA20>MA50)")
    elif price > ma20: b += 6
    elif price < ma20 and price > ma50: b += 2
    if price < ma20 < ma50: be += 14; rbe.append("空头排列")
    elif price < ma20: be += 6
    if last.get("MA_GoldenCross", False): b += 8; rb.append("MA金叉")
    if last.get("MA_DeathCross", False): be += 8; rbe.append("MA死叉")

    # ── Bollinger (15pts) ──
    bu, bm, bl = last["BB_Upper"], last["BB_Middle"], last["BB_Lower"]
    if price < bl * 1.02: b += 14; rb.append("BB下轨超跌⭐")
    elif price < bm and price > prev["Close"]: b += 6
    elif price < bm: b += 3
    if price > bu * 0.98: be += 14; rbe.append("BB上轨见顶⭐")
    elif price > bm: be += 3

    # ── ADX (10pts) ──
    adx, pdi, mdi = last["ADX"], last["DI_Plus"], last["DI_Minus"]
    if adx > 25:
        if pdi > mdi: b += 10; rb.append(f"强趋势↑(ADX{adx:.0f})")
        else: be += 10; rbe.append(f"强趋势↓(ADX{adx:.0f})")
    elif adx > 18:
        if pdi > mdi: b += 5
        else: be += 5

    # ── KDJ (5pts) ──
    k_val, d_val, j_val = last["KDJ_K"], last["KDJ_D"], last["KDJ_J"]
    pk, pd_ = prev["KDJ_K"], prev["KDJ_D"]
    if (k_val > d_val and pk <= pd_) and j_val < 30:
        b += 5; rb.append(f"KDJ金叉(j={j_val:.0f})")
    if (k_val < d_val and pk >= pd_) and j_val > 70:
        be += 5; rbe.append(f"KDJ死叉(j={j_val:.0f})")

    # ── Volume (5pts) ──
    if vol_ratio > 2:
        if b > be: b += 5; rb.append(f"放量({vol_ratio:.1f}x)")
        else: be += 5; rbe.append(f"放量下跌({vol_ratio:.1f}x)")
    elif vol_ratio > 1.3:
        if b > be: b += 3

    # ── Risk metrics ──
    atr_val = tr.iloc[-1] if 'tr' in dir() else price * 0.02
    atr_pct = atr_val / price * 100

    net = b - be
    if abs(net) < 10:
        return None
    conf = min(abs(net) / 100, 0.95)
    direction = "LONG" if net > 0 else "SHORT"

    return {
        "ticker": ticker, "direction": direction, "price": round(price, 2),
        "confidence": round(conf, 2), "net_score": round(net, 0),
        "rsi": round(rsi, 1), "adx": round(adx, 1), "atr_pct": round(atr_pct, 1),
        "vol_ratio": round(vol_ratio, 1), "ma_trend": "多头" if price > ma20 > ma50 else ("空头" if price < ma20 < ma50 else "震荡"),
        "reasons_bull": rb[:5], "reasons_bear": rbe[:5],
    }


# ─── Position Allocator (1000万级别) ─────────────────────────────────────────


def allocate(signals, capital, max_positions, max_single_pct):
    longs = sorted([s for s in signals if s['direction'] == 'LONG'],
                   key=lambda s: s['confidence'], reverse=True)[:max_positions]
    if not longs:
        return [], 0

    # Step 1: confidence-weighted allocation
    total_conf = sum(s['confidence'] for s in longs)
    for s in longs:
        s['raw_weight'] = s['confidence'] / total_conf if total_conf > 0 else 1/len(longs)

    # Step 2: cap single position
    for s in longs:
        s['weight'] = min(s['raw_weight'], max_single_pct)
    # Step 3: renormalize
    tw = sum(s['weight'] for s in longs)
    if tw > 1.0:
        for s in longs:
            s['weight'] /= tw
    # Step 4: allocate remaining to cash
    total_w = sum(s['weight'] for s in longs)
    for s in longs:
        s['amount'] = capital * s['weight']
        s['shares'] = int(s['amount'] / s['price'])
        s['cost'] = s['shares'] * s['price']

    return longs, sum(s['cost'] for s in longs)


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    has_tiingo = bool(os.getenv("TIINGO_API_KEY", ""))

    print("=" * 80)
    print("  📈 AmericanQcode — ¥10,000,000 持仓推荐")
    print(f"  📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  数据: Tiingo + 全套指标")
    print(f"  🎯 最多 {MAX_POSITIONS} 仓位  |  单票 ≤ {MAX_SINGLE_PCT*100:.0f}%  |  保留现金缓冲")
    print("=" * 80)

    if not has_tiingo:
        print("  ⚠️  未检测到 Tiingo Key，请设置 TIINGO_API_KEY")
        return

    print(f"\n🔍 正在扫描 {len(WATCHLIST)} 只标的...\n")
    results = []

    for ticker in WATCHLIST:
        sys.stdout.write(f"  {ticker:<6} ... ")
        sys.stdout.flush()

        df = fetch_tiingo(ticker)
        if df is None or len(df) < 50:
            print("❌")
            continue

        df = compute_indicators(df)
        signal = score_ticker(df, ticker)
        if signal is None:
            print("⚪ 无信号")
            continue

        results.append(signal)
        icon = "🟢" if signal['direction'] == 'LONG' else "🔴"
        print(f"{icon} ${signal['price']:.2f}  "
              f"得分{signal['net_score']:+.0f}  RSI{signal['rsi']:.0f}  "
              f"ADX{signal['adx']:.0f}  {signal['ma_trend']}")
        for r in signal['reasons_bull'][:2]:
            print(f"     ✅ {r}")
        for r in signal['reasons_bear'][:1]:
            print(f"     ⚠️  {r}")

    if not results:
        print("\n❌ 无结果")
        return

    longs = [r for r in results if r['direction'] == 'LONG']
    shorts = [r for r in results if r['direction'] == 'SHORT']

    allocations, invested = allocate(results, CAPITAL, MAX_POSITIONS, MAX_SINGLE_PCT)
    cash = CAPITAL - invested

    print(f"\n{'=' * 80}")
    print(f"  📊 扫描: {len(results)} 信号 | 🟢 {len(longs)} 看多 | 🔴 {len(shorts)} 看空")
    print(f"{'=' * 80}")

    if allocations:
        print(f"\n  🎯 ¥{CAPITAL:,.0f} 推荐持仓\n")
        print(f"  {'标的':<8} {'现价':>8} {'得分':>5} {'RSI':>5} {'ADX':>5} {'趋势':>6} {'ATR%':>6} {'仓位%':>7} {'金额':>14} {'股数':>8}")
        print(f"  {'-' * 78}")

        for a in allocations:
            print(f"  🟢 {a['ticker']:<6} ¥{a['price']:>7,.0f} {a['net_score']:>+4.0f} "
                  f"{a['rsi']:>4.0f} {a['adx']:>4.0f} {a['ma_trend']:<6} "
                  f"{a['atr_pct']:>5.1f}% {a['weight']*100:>6.1f}% ¥{a['amount']:>12,.0f} {a['shares']:>8,}")

        print(f"  {'-' * 78}")
        print(f"  {'合计':>8} {'':>6} {'':>5} {'':>5} {'':>4} {'':>6} {'':>7} {sum(a['weight'] for a in allocations)*100:>6.1f}% ¥{invested:>12,.0f}")
        print(f"\n  💵 现金: ¥{cash:,.0f} ({(cash/CAPITAL*100):.1f}%)")

        print(f"\n  📋 持仓逻辑:\n")
        for a in allocations:
            atr_amount = a['price'] * a['atr_pct'] / 100
            print(f"  [{a['ticker']}] ¥{a['amount']:,.0f} ({a['weight']*100:.1f}%) | "
                  f"{a['shares']:,}股 @ ¥{a['price']:,.0f}")
            print(f"     RSI{a['rsi']:.0f} ADX{a['adx']:.0f} {a['ma_trend']} | "
                  f"日均波动 ¥{atr_amount:.1f}/股 (ATR{a['atr_pct']:.1f}%)")
            if a['reasons_bull']:
                print(f"     ✅ {' | '.join(a['reasons_bull'][:4])}")
            if a['reasons_bear']:
                print(f"     ⚠️  {' | '.join(a['reasons_bear'][:3])}")
            print()

    # Risk Management Summary
    if allocations:
        print(f"  {'=' * 80}")
        print(f"  🛡️  风控建议")
        print(f"  {'=' * 80}")
        print(f"  单票最大: {MAX_SINGLE_PCT*100:.0f}% (¥{CAPITAL*MAX_SINGLE_PCT:,.0f})")
        print(f"  总仓位: {(invested/CAPITAL*100):.1f}%")
        print(f"  现金缓冲: {(cash/CAPITAL*100):.1f}%")
        atr_weighted = sum(a['atr_pct'] * a['weight'] for a in allocations)
        print(f"  组合加权ATR: {atr_weighted:.1f}% (日均波动约 ¥{CAPITAL*atr_weighted/100:,.0f})")
        print(f"  推荐止损: 单票ATR×3 或 -8%硬止损")

    # Bearish warnings
    if shorts:
        print(f"\n  {'=' * 80}")
        print(f"  ⚠️  当前不宜做多")
        print(f"  {'=' * 80}")
        for s in sorted(shorts, key=lambda x: x['confidence'], reverse=True)[:5]:
            print(f"  🔴 {s['ticker']:<6} ¥{s['price']:>8,.0f}  "
                  f"RSI{s['rsi']:.0f}  ADX{s['adx']:.0f}  {s['ma_trend']}")
            for r in s['reasons_bear'][:3]:
                print(f"     ⚠️  {r}")

    print(f"\n  {'=' * 80}")
    print(f"  ⚠️  算法信号仅供参考，不构成投资建议。过往回测不代表未来收益。")
    print(f"  📊 数据: Tiingo 复权价格 | 指标: MACD/RSI/MA/BB/ADX/KDJ 加权评分")
    print(f"  {'=' * 80}")


if __name__ == "__main__":
    main()
