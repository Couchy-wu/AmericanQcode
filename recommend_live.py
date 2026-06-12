#!/usr/bin/env python3
"""
Live position recommendation — $10,000 capital allocation.
Multi-source: Finnhub → iTick → Finviz (auto fallback).

Setup:
  1. Register at https://finnhub.io/register (free, 60 calls/min)
  2. Register at https://itick.org (free, unlimited calls)
  3. Set FINNHUB_API_KEY and/or ITICK_API_TOKEN in .env

With K-line data: full MACD/BB/KDJ/ADX calculation
Without: Finviz snapshot-based scoring

Usage:
  python3 recommend_live.py
"""

import os
import sys
import time
import math
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd

# Load .env
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from finvizfinance.quote import finvizfinance

# ─── Config ──────────────────────────────────────────────────────────────────

CAPITAL = 10_000.0
MAX_POSITIONS = 5
MIN_CONFIDENCE = 0.50
LOOKBACK_DAYS = 200

WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA",
    "AMD", "INTC", "CRM", "ADBE", "NFLX", "DIS", "PYPL", "UBER",
    "SPY", "QQQ", "IWM",
]

# ─── Data Providers ──────────────────────────────────────────────────────────


def fetch_kline_finnhub(ticker: str) -> Optional[pd.DataFrame]:
    """Try Finnhub first (60 calls/min free, historical OHLCV)."""
    api_key = os.getenv("FINNHUB_API_KEY", "")
    if not api_key:
        return None
    try:
        import finnhub
        client = finnhub.Client(api_key=api_key)
        end = int(time.time())
        start = end - LOOKBACK_DAYS * 24 * 3600
        resp = client.stock_candles(ticker.upper(), "D", start, end)
        if resp.get("s") != "ok" or not resp.get("c"):
            return None
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(resp["t"], unit="s"),
            "Open": resp["o"], "High": resp["h"],
            "Low": resp["l"], "Close": resp["c"], "Volume": resp["v"],
        }).set_index("timestamp").sort_index()
        return df if len(df) >= 50 else None
    except Exception:
        return None


def fetch_kline_itick(ticker: str) -> Optional[pd.DataFrame]:
    """Try iTick/AllTick (unlimited free calls, global markets)."""
    api_token = os.getenv("ITICK_API_TOKEN", "")
    if not api_token:
        return None
    try:
        import json, urllib.parse, requests
        query = {
            "trace": "recommend",
            "data": {
                "code": f"{ticker.upper()}.US",
                "kline_type": 8,         # 8 = daily
                "kline_timestamp_end": 0,
                "query_kline_num": 250,
                "adjust_type": 0,
            },
        }
        encoded = urllib.parse.quote(json.dumps(query))
        url = f"https://quote.alltick.io/quote-stock-b-api/kline?token={api_token}&query={encoded}"
        resp = requests.get(url, headers={"Content-Type": "application/json"}, timeout=15)
        data = resp.json()
        if data.get("code") != 200:
            return None
        klines = data["data"].get("kline_list", [])
        if not klines:
            return None
        rows = []
        for k in klines:
            if len(k) >= 6:
                ts = pd.Timestamp(k[0], unit="ms") if k[0] > 1e12 else pd.Timestamp(k[0], unit="s")
                rows.append({"timestamp": ts, "Open": float(k[1]), "High": float(k[2]),
                            "Low": float(k[3]), "Close": float(k[4]), "Volume": float(k[5] or 0)})
        if not rows:
            return None
        return pd.DataFrame(rows).set_index("timestamp").sort_index()
    except Exception:
        return None


def fetch_kline(ticker: str) -> tuple[Optional[pd.DataFrame], str]:
    """Try all K-line providers in order. Returns (df, source_name)."""
    for fetcher, name in [(fetch_kline_finnhub, "Finnhub"), (fetch_kline_itick, "iTick")]:
        df = fetcher(ticker)
        if df is not None and len(df) >= 50:
            return df, name
    return None, "none"


# ─── Indicators (pure pandas/numpy) ──────────────────────────────────────────


def ema(s: pd.Series, p: int) -> pd.Series:
    return s.ewm(span=p, adjust=False).mean()

def sma(s: pd.Series, p: int) -> pd.Series:
    return s.rolling(p).mean()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Full indicator suite: MACD, RSI, MA, BB, ADX, KDJ."""
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    # MACD
    ml = ema(c, 12) - ema(c, 26)
    sl = ema(ml, 9)
    df["MACD"], df["MACD_Signal"], df["MACD_Histogram"] = ml, sl, ml - sl
    df["MACD_GoldenCross"] = (ml > sl) & (ml.shift(1) <= sl.shift(1))
    df["MACD_DeathCross"] = (ml < sl) & (ml.shift(1) >= sl.shift(1))

    # RSI
    d = c.diff()
    g = d.where(d > 0, 0.0)
    lo = (-d).where(d < 0, 0.0)
    rs = g.ewm(alpha=1/14, adjust=False).mean() / lo.ewm(alpha=1/14, adjust=False).mean()
    df["RSI"] = 100 - 100 / (1 + rs)
    df["RSI_ExitOversold"] = (df["RSI"] > 30) & (df["RSI"].shift(1) <= 30)
    df["RSI_ExitOverbought"] = (df["RSI"] < 70) & (df["RSI"].shift(1) >= 70)

    # MA
    df["MA_20"] = ema(c, 20)
    df["MA_50"] = ema(c, 50)
    df["MA_GoldenCross"] = (df["MA_20"] > df["MA_50"]) & (df["MA_20"].shift(1) <= df["MA_50"].shift(1))
    df["MA_DeathCross"] = (df["MA_20"] < df["MA_50"]) & (df["MA_20"].shift(1) >= df["MA_50"].shift(1))

    # Bollinger
    bb_mid = sma(c, 20)
    bb_std = c.rolling(20).std()
    df["BB_Upper"] = bb_mid + 2 * bb_std
    df["BB_Middle"] = bb_mid
    df["BB_Lower"] = bb_mid - 2 * bb_std

    # ADX
    tr1, tr2, tr3 = h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    um, dm = h - h.shift(1), l.shift(1) - l
    pdi = 100 * pd.Series(np.where((um > dm) & (um > 0), um, 0), index=df.index).ewm(alpha=1/14, adjust=False).mean() / atr
    mdi = 100 * pd.Series(np.where((dm > um) & (dm > 0), dm, 0), index=df.index).ewm(alpha=1/14, adjust=False).mean() / atr
    df["ADX"] = (100 * (pdi - mdi).abs() / (pdi + mdi + 1e-10)).ewm(alpha=1/14, adjust=False).mean()
    df["DI_Plus"], df["DI_Minus"] = pdi, mdi

    # KDJ
    ll, hh = l.rolling(9).min(), h.rolling(9).max()
    rsv = (c - ll) / (hh - ll + 1e-10) * 100
    k = rsv.ewm(alpha=1/3, adjust=False).mean()
    d2 = k.ewm(alpha=1/3, adjust=False).mean()
    df["KDJ_K"], df["KDJ_D"], df["KDJ_J"] = k, d2, 3*k - 2*d2

    return df


# ─── Full K-line based scoring ───────────────────────────────────────────────


def score_ticker_full(df: pd.DataFrame, ticker: str) -> Optional[dict]:
    """Full multi-indicator scoring with historical K-line data."""
    if len(df) < 50:
        return None

    last = df.iloc[-1]
    prev = df.iloc[-2]
    price = float(last["Close"])
    avg_vol = float(df["Volume"].rolling(20).mean().iloc[-1])
    vol = float(last.get("Volume", 0))
    vol_ratio = vol / avg_vol if avg_vol > 0 else 1.0

    bullish = 0.0
    bearish = 0.0
    rb, rbe = [], []

    # ── MACD (25pts) ──
    if last.get("MACD_GoldenCross", False):
        bullish += 25; rb.append("MACD金叉⭐")
    elif last["MACD"] > last["MACD_Signal"]:
        bullish += 12
        if last["MACD_Histogram"] > prev.get("MACD_Histogram", 0):
            bullish += 5; rb.append("MACD多头增强")
    if last.get("MACD_DeathCross", False):
        bearish += 25; rbe.append("MACD死叉⭐")
    elif last["MACD"] < last["MACD_Signal"]:
        bearish += 12

    # ── RSI (20pts) ──
    rsi = last.get("RSI", 50)
    if rsi < 30:
        bullish += 18; rb.append(f"RSI超卖({rsi:.0f})⭐")
    elif rsi < 40 and rsi > prev.get("RSI", 50):
        bullish += 8; rb.append(f"RSI低位反弹({rsi:.0f})")
    if rsi > 70:
        bearish += 18; rbe.append(f"RSI超买({rsi:.0f})⭐")
    elif rsi > 60 and rsi < prev.get("RSI", 50):
        bearish += 8; rbe.append(f"RSI高位回落({rsi:.0f})")

    # RSI divergence
    if len(df) >= 20:
        p1, p2 = df["Close"].iloc[-20:-10], df["Close"].iloc[-10:]
        r1, r2 = df["RSI"].iloc[-20:-10], df["RSI"].iloc[-10:]
        if p2.min() < p1.min() and r2.min() > r1.min():
            bullish += 20; rb.append("RSI看涨背离⭐⭐")
        if p2.max() > p1.max() and r2.max() < r1.max():
            bearish += 20; rbe.append("RSI看跌背离⭐⭐")

    # ── MA Trend (20pts) ──
    ma20, ma50 = last["MA_20"], last["MA_50"]
    if price > ma20 > ma50:
        bullish += 14; rb.append("多头排列")
    elif price > ma20:
        bullish += 6
    if price < ma20 < ma50:
        bearish += 14; rbe.append("空头排列")
    if last.get("MA_GoldenCross", False):
        bullish += 8; rb.append("MA金叉")
    if last.get("MA_DeathCross", False):
        bearish += 8; rbe.append("MA死叉")

    # ── Bollinger (15pts) ──
    bu, bm, bl = last["BB_Upper"], last["BB_Middle"], last["BB_Lower"]
    if price < bl * 1.02:
        bullish += 14; rb.append("BB下轨超跌⭐")
    elif price < bm and price > prev["Close"]:
        bullish += 6
    if price > bu * 0.98:
        bearish += 14; rbe.append("BB上轨见顶⭐")

    # ── ADX (10pts) ──
    adx, pdi, mdi = last["ADX"], last["DI_Plus"], last["DI_Minus"]
    if adx > 25:
        if pdi > mdi:
            bullish += 10; rb.append(f"强趋势↑(ADX={adx:.0f})")
        else:
            bearish += 10; rbe.append(f"强趋势↓(ADX={adx:.0f})")
    elif adx > 18:
        if pdi > mdi: bullish += 5
        else: bearish += 5

    # ── KDJ (5pts) ──
    k_val, d_val, j_val = last["KDJ_K"], last["KDJ_D"], last["KDJ_J"]
    pk, pd_ = prev["KDJ_K"], prev["KDJ_D"]
    if (k_val > d_val and pk <= pd_) and j_val < 30:
        bullish += 5; rb.append(f"KDJ金叉(j={j_val:.0f})")
    if (k_val < d_val and pk >= pd_) and j_val > 70:
        bearish += 5; rbe.append(f"KDJ死叉(j={j_val:.0f})")

    # ── Volume (5pts) ──
    if vol_ratio > 2:
        if bullish > bearish: bullish += 5
        else: bearish += 5
        rb.append(f"巨量({vol_ratio:.1f}x)")

    net = bullish - bearish
    conf = min(abs(net) / 100, 0.95)
    direction = "LONG" if net > 0 else "SHORT"

    if abs(net) < 15:
        return None  # Too weak

    return {
        "ticker": ticker, "direction": direction, "price": round(price, 2),
        "confidence": round(conf, 2), "net_score": round(net, 0),
        "rsi": round(rsi, 1), "adx": round(adx, 1), "vol_ratio": round(vol_ratio, 1),
        "reasons_bull": rb[:5], "reasons_bear": rbe[:5],
        "momentum": 0, "rel_volume": vol_ratio, "target_upside": 0,
    }


# ─── Finviz snapshot fallback ────────────────────────────────────────────────


def safe_float(val, default=0.0):
    if val is None or val == '-' or val == '' or val == 'N/A':
        return default
    if isinstance(val, (int, float)):
        return float(val)
    try:
        return float(str(val).replace(',', '').replace('%', '').replace('$', '').replace('B', '').strip())
    except (ValueError, TypeError):
        return default


def score_ticker_finviz(info: dict, ticker: str) -> Optional[dict]:
    """Snapshot-based scoring when no K-line data is available."""
    price = safe_float(info.get('Price'))
    if price <= 0:
        return None

    rsi = safe_float(info.get('RSI (14)'), 50)
    sma20_pct = safe_float(info.get('SMA20'), 0)
    sma50_pct = safe_float(info.get('SMA50'), 0)
    sma200_pct = safe_float(info.get('SMA200'), 0)
    rel_volume = safe_float(info.get('Rel Volume'), 1.0)
    recom = safe_float(info.get('Recom'), 2.5)
    target_price = safe_float(info.get('Target Price'), 0)
    perf_week = safe_float(info.get('Perf Week'), 0)
    perf_month = safe_float(info.get('Perf Month'), 0)
    perf_quarter = safe_float(info.get('Perf Quarter'), 0)
    perf_half = safe_float(info.get('Perf Half Y'), 0)
    peg = safe_float(info.get('PEG'), 0)
    short_float = safe_float(info.get('Short Float'), 0)
    beta = safe_float(info.get('Beta'), 1.0)
    atr = safe_float(info.get('ATR (14)'), 0)
    volatility_w = safe_float(info.get('Volatility W'), 0)
    change_pct = safe_float(info.get('Change'), 0)

    momentum = perf_week * 0.15 + perf_month * 0.25 + perf_quarter * 0.35 + perf_half * 0.15

    bullish, bearish = 0.0, 0.0
    rb, rbe = [], []

    # RSI
    if rsi < 30: bullish += 20; rb.append(f"RSI超卖({rsi:.0f})⭐")
    elif rsi < 40: bullish += 12; rb.append(f"RSI低位({rsi:.0f})")
    elif rsi > 70: bearish += 20; rbe.append(f"RSI超买({rsi:.0f})⭐")
    elif rsi > 60: bearish += 10

    # SMA trend
    if sma200_pct > 10: bullish += 8; rb.append(f"价>SMA200+{sma200_pct:.0f}%")
    if sma50_pct > 5: bullish += 8
    if sma20_pct > 3: bullish += 6
    if sma200_pct < -10: bearish += 8; rbe.append(f"价<SMA200{sma200_pct:.0f}%")
    if sma50_pct < -5: bearish += 8
    if sma20_pct < -3: bearish += 6

    # Momentum
    if momentum > 5: bullish += 15; rb.append(f"强动量({momentum:.0f}%)")
    elif momentum > 2: bullish += 8
    elif momentum < -5: bearish += 15; rbe.append(f"弱动量({momentum:.0f}%)")

    # Volume
    if rel_volume > 2:
        if momentum > 0: bullish += 12
        else: bearish += 12
    elif rel_volume > 1.3:
        if momentum > 0: bullish += 6
        else: bearish += 6

    # Analyst
    if recom <= 1.5: bullish += 8; rb.append(f"分析师强力推荐({recom:.1f})")
    elif recom >= 4.0: bearish += 5; rbe.append(f"分析师看空({recom:.1f})")
    if target_price > 0 and price > 0:
        upside = (target_price / price - 1) * 100
        if upside > 15: bullish += 6; rb.append(f"目标价+{upside:.0f}%")
        elif upside < -10: bearish += 4

    # BB estimate from SMA20 + volatility
    if abs(sma20_pct) > 5 and volatility_w > 3:
        if sma20_pct < 0: bullish += 5; rb.append("BB下轨区(估计)")
        else: bearish += 5; rbe.append("BB上轨区(估计)")

    # PEG
    if 0 < peg < 1.5: bullish += 4
    elif peg > 3: bearish += 2

    # Short squeeze potential
    if short_float > 10 and momentum > 0: bullish += 4; rb.append(f"高做空率{short_float:.0f}%→逼空")
    elif short_float > 10: bearish += 4; rbe.append(f"高做空率{short_float:.0f}%→看空")

    net = bullish - bearish
    if abs(net) < 10:
        return None

    conf = min(abs(net) / 100, 0.95)
    direction = "LONG" if net > 0 else "SHORT"

    return {
        "ticker": ticker, "direction": direction, "price": round(price, 2),
        "confidence": round(conf, 2), "net_score": round(net, 0),
        "rsi": round(rsi, 1), "momentum": round(momentum, 1),
        "rel_volume": rel_volume, "target_upside": round((target_price / price - 1) * 100, 1) if target_price > 0 else 0,
        "reasons_bull": rb[:5], "reasons_bear": rbe[:5],
        "adx": 0, "vol_ratio": rel_volume,
    }


# ─── Position Allocator ──────────────────────────────────────────────────────


def allocate(signals, capital, max_positions):
    longs = sorted(
        [s for s in signals if s['direction'] == 'LONG'],
        key=lambda s: s['confidence'], reverse=True
    )[:max_positions]

    if not longs:
        return [], 0

    total_conf = sum(s['confidence'] for s in longs)
    if total_conf == 0:
        return [], 0

    allocations = []
    for s in longs:
        w = min(s['confidence'] / total_conf, 0.35)
        amt = capital * w
        shares = int(amt / s['price'])
        allocations.append({**s, 'weight': round(w * 100, 1), 'amount': round(amt, 2),
                           'shares': shares, 'cost': round(shares * s['price'], 2)})

    total_cost = sum(a['cost'] for a in allocations)
    if total_cost > capital:
        scale = capital / total_cost
        for a in allocations:
            a['amount'] = round(a['amount'] * scale, 2)
            a['shares'] = int(a['amount'] / a['price'])
            a['cost'] = round(a['shares'] * a['price'], 2)
            a['weight'] = round(a['cost'] / capital * 100, 1)

    return allocations, sum(a['cost'] for a in allocations)


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    has_finnhub = bool(os.getenv("FINNHUB_API_KEY"))
    has_itick = bool(os.getenv("ITICK_API_TOKEN"))
    kline_available = has_finnhub or has_itick

    print("=" * 90)
    print("  📈 AmericanQcode — 实时持仓推荐")
    print(f"  💰 资金: ${CAPITAL:,.0f}  |  📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  🎯 最多 {MAX_POSITIONS} 仓位")
    print(f"  📊 数据源: ", end="")
    sources = []
    if has_finnhub: sources.append("Finnhub (K线)")
    if has_itick: sources.append("iTick (K线)")
    sources.append("Finviz (快照)")
    print(" → ".join(sources))
    if not kline_available:
        print("\n  💡 提示: 设置 FINNHUB_API_KEY 或 ITICK_API_TOKEN 环境变量")
        print("     即可获取完整K线数据并计算 MACD/布林带/KDJ 等全部指标")
        print("     免费注册: https://finnhub.io/register 或 https://itick.org\n")
    print("=" * 90)

    print("\n🔍 扫描股票池...\n")
    results = []
    kline_count = 0
    finviz_count = 0

    for ticker in WATCHLIST:
        sys.stdout.write(f"  {ticker:<6} ... ")
        sys.stdout.flush()

        # Try K-line providers first
        df, source = fetch_kline(ticker)
        if df is not None and len(df) >= 50:
            df = compute_indicators(df)
            signal = score_ticker_full(df, ticker)
            if signal:
                results.append(signal)
                kline_count += 1
                icon = "🟢" if signal['direction'] == 'LONG' else "🔴"
                print(f"{icon} ${signal['price']:.2f}  得分{signal['net_score']:+.0f}  "
                      f"RSI{signal['rsi']:.0f}  ADX{signal.get('adx',0):.0f}  [{source}K线]")
                for r in signal['reasons_bull'][:2] + signal['reasons_bear'][:2]:
                    pfx = "   ✅" if r in signal['reasons_bull'] else "   ⚠️ "
                    print(f"{pfx} {r}")
                time.sleep(0.5)
                continue

        # Fallback to Finviz snapshot
        try:
            info = finvizfinance(ticker).ticker_fundament()
            if info and info.get('Price'):
                signal = score_ticker_finviz(info, ticker)
                if signal:
                    results.append(signal)
                    finviz_count += 1
                    icon = "🟢" if signal['direction'] == 'LONG' else "🔴"
                    print(f"{icon} ${signal['price']:.2f}  得分{signal['net_score']:+.0f}  "
                          f"RSI{signal['rsi']:.0f}  [Finviz快照]")
                    for r in signal['reasons_bull'][:2] + signal['reasons_bear'][:2]:
                        pfx = "   ✅" if r in signal['reasons_bull'] else "   ⚠️ "
                        print(f"{pfx} {r}")
                else:
                    print("⚪ 无明确信号")
            else:
                print("❌ 无数据")
        except Exception:
            print("❌")
        time.sleep(1.5)

    if not results:
        print("\n❌ 无法获取数据。请检查网络或设置 API Key。")
        return

    # Allocate
    allocations, invested = allocate(results, CAPITAL, MAX_POSITIONS)
    longs = [r for r in results if r['direction'] == 'LONG']
    shorts = [r for r in results if r['direction'] == 'SHORT']

    print(f"\n{'=' * 90}")
    print(f"  📊 扫描摘要")
    print(f"{'=' * 90}")
    print(f"  总计: {len(results)} 信号  |  🟢 看多: {len(longs)}  |  🔴 看空: {len(shorts)}")
    print(f"  K线数据: {kline_count} 只  |  Finviz快照: {finviz_count} 只")
    if kline_count > 0:
        print(f"  含MACD/BB/KDJ/ADX完整指标: {kline_count} 只 ✅")

    if allocations:
        print(f"\n{'=' * 90}")
        print(f"  🎯 ${CAPITAL:,} 推荐持仓")
        print(f"{'=' * 90}")
        print(f"\n{'标的':<8} {'价格':>8} {'RSI':>5} {'得分':>6} {'置信度':>7} {'指标':<12} {'权重%':>7} {'金额':>10} {'股数':>6}")
        print("-" * 89)

        for a in allocations:
            ind_detail = ""
            if a.get("adx", 0) > 0:
                ind_detail = f"ADX{a['adx']:.0f}"
            elif a.get("momentum", 0) != 0:
                ind_detail = f"动量{a['momentum']:+.0f}%"
            print(f"🟢 {a['ticker']:<6} ${a['price']:>7.2f} {a['rsi']:>4.0f} "
                  f"{a['net_score']:>+5.0f} {a['confidence']:>6.0%} {ind_detail:<12} "
                  f"{a['weight']:>6.1f}% ${a['amount']:>9,.0f} {a['shares']:>6}")

        print("-" * 89)
        tw = sum(a['weight'] for a in allocations)
        print(f"{'合计':>8} {'':>6} {'':>5} {'':>6} {'':>7} {'':<12} {tw:>6.1f}% ${invested:>9,.0f}")
        cash = CAPITAL - invested
        print(f"\n💵 剩余现金: ${cash:,.0f} ({(cash/CAPITAL*100):.1f}%)")

        print("\n📋 每笔详解:")
        for a in allocations:
            print(f"\n  🟢 [{a['ticker']}] ${a['amount']:,.0f} ({a['weight']:.1f}%) | "
                  f"{a['shares']}股 @ ${a['price']:.2f} | 置信度: {a['confidence']:.0%}")
            if a['reasons_bull']:
                print(f"     ✅ 看多: {' | '.join(a['reasons_bull'][:5])}")
            if a['reasons_bear']:
                print(f"     ⚠️  风险: {' | '.join(a['reasons_bear'][:3])}")

    if shorts:
        print(f"\n{'=' * 90}")
        print(f"  ⚠️  看空预警")
        print(f"{'=' * 90}")
        for s in sorted(shorts, key=lambda x: x['confidence'], reverse=True)[:5]:
            print(f"\n  🔴 {s['ticker']:<6} ${s['price']:.2f}  RSI{s['rsi']:.0f}  "
                  f"得分{s['net_score']:+.0f}  置信度{s['confidence']:.0%}")
            for r in s['reasons_bear'][:3]:
                print(f"     ⚠️  {r}")

    print(f"\n{'=' * 90}")
    print(f"  ⚠️  免责声明: 算法信号仅供参考，不构成投资建议。")
    print(f"  📊 数据源: {' → '.join(sources)}")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
