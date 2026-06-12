#!/usr/bin/env python3
"""
Live position recommendation — $10,000 capital allocation.
Scans current market and recommends buy/sell with position sizing.
"""

import warnings
from datetime import datetime, timedelta
from typing import Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ─── Config ──────────────────────────────────────────────────────────────────

CAPITAL = 10_000.0
MAX_POSITIONS = 5                    # Max concurrent positions
POSITION_PCT = 0.20                  # 20% per position ($2,000)
MIN_CONFIDENCE = 0.55                # Only recommend signals above this
LOOKBACK_DAYS = 200                  # Need enough bars for indicator calculation

# Broad watchlist — high liquidity US stocks
WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA",
    "AMD", "INTC", "CRM", "ADBE", "NFLX", "DIS", "PYPL", "UBER",
    "SPY", "QQQ", "IWM", "DIA",
]

# ─── Indicator Functions (pure pandas/numpy) ─────────────────────────────────


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all required indicators."""
    close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]

    # MACD
    macd_line = ema(close, 12) - ema(close, 26)
    signal_line = ema(macd_line, 9)
    df["MACD"] = macd_line
    df["MACD_Signal"] = signal_line
    df["MACD_Histogram"] = macd_line - signal_line
    df["MACD_GoldenCross"] = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
    df["MACD_DeathCross"] = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100.0 - (100.0 / (1.0 + rs))
    df["RSI_Oversold"] = df["RSI"] < 30
    df["RSI_Overbought"] = df["RSI"] > 70
    df["RSI_ExitOversold"] = (df["RSI"] > 30) & (df["RSI"].shift(1) <= 30)
    df["RSI_ExitOverbought"] = (df["RSI"] < 70) & (df["RSI"].shift(1) >= 70)

    # MA
    df["MA_20"] = ema(close, 20)
    df["MA_50"] = ema(close, 50)
    df["MA_GoldenCross"] = (df["MA_20"] > df["MA_50"]) & (df["MA_20"].shift(1) <= df["MA_50"].shift(1))
    df["MA_DeathCross"] = (df["MA_20"] < df["MA_50"]) & (df["MA_20"].shift(1) >= df["MA_50"].shift(1))

    # Bollinger Bands
    bb_mid = sma(close, 20)
    bb_std = close.rolling(20).std()
    df["BB_Upper"] = bb_mid + 2 * bb_std
    df["BB_Middle"] = bb_mid
    df["BB_Lower"] = bb_mid - 2 * bb_std
    df["BB_Bandwidth"] = (df["BB_Upper"] - df["BB_Lower"]) / bb_mid
    df["BB_Squeeze"] = df["BB_Bandwidth"] < df["BB_Bandwidth"].rolling(100).quantile(0.1)

    # ADX
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / atr
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    df["ADX"] = dx.ewm(alpha=1/14, adjust=False).mean()
    df["DI_Plus"] = plus_di
    df["DI_Minus"] = minus_di

    # KDJ
    lowest = low.rolling(9).min()
    highest = high.rolling(9).max()
    rsv = (close - lowest) / (highest - lowest + 1e-10) * 100
    k = rsv.ewm(alpha=1/3, adjust=False).mean()
    d = k.ewm(alpha=1/3, adjust=False).mean()
    df["KDJ_K"] = k
    df["KDJ_D"] = d
    df["KDJ_J"] = 3 * k - 2 * d

    return df


# ─── Combined Scoring Engine ─────────────────────────────────────────────────


def score_ticker(df: pd.DataFrame, ticker: str) -> dict:
    """
    Multi-strategy scoring for a single ticker.
    Returns a signal with direction, strength, and reasoning breakdown.
    """
    if len(df) < 50:
        return None

    last = df.iloc[-1]
    prev = df.iloc[-2]
    price = float(last["Close"])
    avg_vol = float(df["Volume"].rolling(20).mean().iloc[-1])
    vol = float(last.get("Volume", 0))
    vol_ratio = vol / avg_vol if avg_vol > 0 else 1.0

    bullish_score = 0.0
    bearish_score = 0.0
    reasons_bull = []
    reasons_bear = []

    # ── 1. MACD 金叉/死叉 (权重: 25) ──
    macd = last.get("MACD", 0)
    macd_sig = last.get("MACD_Signal", 0)
    hist = last.get("MACD_Histogram", 0)
    prev_hist = prev.get("MACD_Histogram", 0)

    if last.get("MACD_GoldenCross", False):
        bullish_score += 25
        reasons_bull.append("MACD金叉")
    elif macd > macd_sig:
        bullish_score += 12
        if hist > prev_hist:
            bullish_score += 5
            reasons_bull.append("MACD多头增强")

    if last.get("MACD_DeathCross", False):
        bearish_score += 25
        reasons_bear.append("MACD死叉")
    elif macd < macd_sig:
        bearish_score += 12

    # ── 2. RSI 位置 (权重: 20) ──
    rsi = last.get("RSI", 50)
    prev_rsi = prev.get("RSI", 50)

    if rsi < 30:
        bullish_score += 15
        reasons_bull.append(f"RSI超卖({rsi:.0f})")
    elif rsi < 40 and rsi > prev_rsi:
        bullish_score += 8
        reasons_bull.append(f"RSI低位反弹({rsi:.0f})")
    elif rsi < 50 and rsi > prev_rsi:
        bullish_score += 5

    if rsi > 70:
        bearish_score += 15
        reasons_bear.append(f"RSI超买({rsi:.0f})")
    elif rsi > 60 and rsi < prev_rsi:
        bearish_score += 8
        reasons_bear.append(f"RSI高位回落({rsi:.0f})")

    # RSI 背离
    if len(df) >= 20:
        rsi_half_early = df["RSI"].iloc[-20:-10].min()
        rsi_half_late = df["RSI"].iloc[-10:].min()
        p_half_early = df["Close"].iloc[-20:-10].min()
        p_half_late = df["Close"].iloc[-10:].min()
        if p_half_late < p_half_early and rsi_half_late > rsi_half_early:
            bullish_score += 20
            reasons_bull.append("RSI看涨背离⭐")
        rsi_max_early = df["RSI"].iloc[-20:-10].max()
        rsi_max_late = df["RSI"].iloc[-10:].max()
        p_max_early = df["Close"].iloc[-20:-10].max()
        p_max_late = df["Close"].iloc[-10:].max()
        if p_max_late > p_max_early and rsi_max_late < rsi_max_early:
            bearish_score += 20
            reasons_bear.append("RSI看跌背离⭐")

    # ── 3. 均线趋势 (权重: 20) ──
    ma20 = last.get("MA_20", price)
    ma50 = last.get("MA_50", price)
    if price > ma20 > ma50:
        bullish_score += 15
        reasons_bull.append("多头排列(价>MA20>MA50)")
    elif price > ma20:
        bullish_score += 8
    if price < ma20 < ma50:
        bearish_score += 15
        reasons_bear.append("空头排列(价<MA20<MA50)")
    elif price < ma20:
        bearish_score += 8

    if last.get("MA_GoldenCross", False):
        bullish_score += 10
        reasons_bull.append("MA金叉")
    if last.get("MA_DeathCross", False):
        bearish_score += 10
        reasons_bear.append("MA死叉")

    # ── 4. 布林带位置 (权重: 15) ──
    bb_upper = last.get("BB_Upper", price * 2)
    bb_mid = last.get("BB_Middle", price)
    bb_lower = last.get("BB_Lower", 0)
    bb_squeeze = last.get("BB_Squeeze", False)

    near_lower = price < bb_lower * 1.02
    near_upper = price > bb_upper * 0.98

    if near_lower:
        bullish_score += 15
        reasons_bull.append("BB下轨超跌")
    elif price < bb_mid and price > prev.get("Close", price) and prev.get("Close", price) < bb_mid:
        bullish_score += 8
        reasons_bull.append("BB反弹")

    if near_upper:
        bearish_score += 15
        reasons_bear.append("BB上轨见顶")
    elif price > bb_mid and price < prev.get("Close", price) and prev.get("Close", price) > bb_mid:
        bearish_score += 8
        reasons_bear.append("BB回落")

    if bb_squeeze:
        if price > bb_mid:
            bullish_score += 8
            reasons_bull.append("BB挤压待突破")
        else:
            bearish_score += 8
            reasons_bear.append("BB挤压待下跌")

    # ── 5. ADX 趋势强度 (权重: 10) ──
    adx = last.get("ADX", 0)
    plus_di = last.get("DI_Plus", 0)
    minus_di = last.get("DI_Minus", 0)

    if adx > 25:
        if plus_di > minus_di:
            bullish_score += 10
            reasons_bull.append(f"强趋势向上(ADX={adx:.0f})")
        else:
            bearish_score += 10
            reasons_bear.append(f"强趋势向下(ADX={adx:.0f})")
    elif adx > 20:
        if plus_di > minus_di:
            bullish_score += 5
        else:
            bearish_score += 5

    # ── 6. KDJ (权重: 5) ──
    k_val = last.get("KDJ_K", 50)
    d_val = last.get("KDJ_D", 50)
    j_val = last.get("KDJ_J", 50)
    prev_k = prev.get("KDJ_K", 50)
    prev_d = prev.get("KDJ_D", 50)

    kdj_golden = (k_val > d_val and prev_k <= prev_d)
    kdj_death = (k_val < d_val and prev_k >= prev_d)

    if kdj_golden and j_val < 30:
        bullish_score += 8
        reasons_bull.append(f"KDJ底部金叉(j={j_val:.0f})")
    elif kdj_golden:
        bullish_score += 4

    if kdj_death and j_val > 70:
        bearish_score += 8
        reasons_bear.append(f"KDJ顶部死叉(j={j_val:.0f})")
    elif kdj_death:
        bearish_score += 4

    # ── 7. 成交量 (权重: 5) ──
    if vol_ratio > 2.0:
        if bullish_score > bearish_score:
            bullish_score += 5
            reasons_bull.append(f"巨量({vol_ratio:.1f}x)")
        else:
            bearish_score += 5
            reasons_bear.append(f"巨量({vol_ratio:.1f}x)")
    elif vol_ratio > 1.5:
        if bullish_score > bearish_score:
            bullish_score += 3

    # ── 最终判定 ──
    total_score = abs(bullish_score - bearish_score)
    max_score = 100.0
    confidence = total_score / max_score

    if bullish_score > bearish_score and confidence >= MIN_CONFIDENCE:
        return {
            "ticker": ticker,
            "direction": "LONG",
            "price": round(price, 2),
            "confidence": round(confidence, 2),
            "score": round(bullish_score - bearish_score, 1),
            "rsi": round(rsi, 1),
            "adx": round(adx, 1),
            "vol_ratio": round(vol_ratio, 1),
            "reasons": reasons_bull[:5],
            "warnings": reasons_bear[:3],
        }
    elif bearish_score > bullish_score and confidence >= MIN_CONFIDENCE:
        return {
            "ticker": ticker,
            "direction": "SHORT",
            "price": round(price, 2),
            "confidence": round(confidence, 2),
            "score": round(bearish_score - bullish_score, 1),
            "rsi": round(rsi, 1),
            "adx": round(adx, 1),
            "vol_ratio": round(vol_ratio, 1),
            "reasons": reasons_bear[:5],
            "warnings": reasons_bull[:3],
        }

    return None


# ─── Data Fetching ────────────────────────────────────────────────────────────


def fetch_data(ticker: str) -> Optional[pd.DataFrame]:
    end = datetime.now()
    start = end - timedelta(days=LOOKBACK_DAYS)
    for attempt in range(3):
        try:
            df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                           end=end.strftime("%Y-%m-%d"), interval="1d",
                           progress=False, auto_adjust=True)
            if not df.empty and len(df) >= 50:
                cols = ["Open", "High", "Low", "Close", "Volume"]
                return df[[c for c in cols if c in df.columns]]
        except Exception:
            import time
            time.sleep(3 * (attempt + 1))
    return None


# ─── Position Sizer ──────────────────────────────────────────────────────────


def allocate_positions(signals: list[dict], capital: float, max_positions: int) -> list[dict]:
    """
    Allocate capital across top signals.
    Kelly-inspired: confidence-weighted allocation with diversification cap.
    """
    if not signals:
        return []

    # Sort by confidence descending
    signals = sorted(signals, key=lambda s: s["confidence"], reverse=True)

    # Take top N signals
    selected = signals[:max_positions]

    # Weight by relative confidence (Kelly fraction)
    total_conf = sum(s["confidence"] for s in selected)
    allocations = []

    for s in selected:
        weight = s["confidence"] / total_conf if total_conf > 0 else 1 / len(selected)
        # Cap individual position at 30% of capital
        weight = min(weight, 0.30)
        alloc = capital * weight
        shares = int(alloc / s["price"])

        allocations.append({
            **s,
            "allocation_pct": round(weight * 100, 1),
            "allocation_usd": round(alloc, 2),
            "shares": shares,
            "cost_basis": round(shares * s["price"], 2),
        })

    # Normalize to ensure total doesn't exceed capital
    total_alloc = sum(a["allocation_usd"] for a in allocations)
    if total_alloc > capital:
        scale = capital / total_alloc
        for a in allocations:
            a["allocation_usd"] = round(a["allocation_usd"] * scale, 2)
            a["shares"] = int(a["allocation_usd"] / a["price"])
            a["cost_basis"] = round(a["shares"] * a["price"], 2)
            a["allocation_pct"] = round(a["allocation_usd"] / capital * 100, 1)

    return allocations


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    print("=" * 90)
    print("  📈 AmericanQcode — 实时持仓推荐")
    print(f"  💰 总资金: ${CAPITAL:,.0f}  |  📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  🎯 最多 {MAX_POSITIONS} 个仓位  |  每仓 {(POSITION_PCT * 100):.0f}% (${CAPITAL * POSITION_PCT:,.0f})")
    print("=" * 90)

    # Fetch data
    print("\n🔍 正在获取市场数据...\n")
    results = {}
    for ticker in WATCHLIST:
        sys.stdout.write(f"  {ticker} ... ")
        sys.stdout.flush()
        df = fetch_data(ticker)
        if df is not None:
            df = compute_all_indicators(df)
            signal = score_ticker(df, ticker)
            if signal:
                results[ticker] = signal
                icon = "🟢" if signal["direction"] == "LONG" else "🔴"
                print(f"{icon} {signal['direction']} (置信度 {signal['confidence']:.0%}, RSI {signal['rsi']:.0f})")
                for r in signal["reasons"][:3]:
                    print(f"     └─ {r}")
            else:
                print("⚪ 无明确信号")
        else:
            print("❌")
        sys.stdout.flush()

    if not results:
        print("\n❌ 无法获取数据，请稍后重试（Yahoo Finance 可能限流中）")
        return

    # Separate LONG and SHORT
    longs = [s for s in results.values() if s["direction"] == "LONG"]
    shorts = [s for s in results.values() if s["direction"] == "SHORT"]

    # Allocate longs only (short selling more complex, skip for simplicity)
    if longs:
        allocations = allocate_positions(longs, CAPITAL, MAX_POSITIONS)
        print("\n" + "=" * 90)
        print("  🎯 推荐持仓 (LONG)")
        print("=" * 90)
        print(f"\n{'标的':<8} {'价格':>8} {'方向':>6} {'得分':>6} {'置信度':>8} {'RSI':>6} {'ADX':>6} {'仓位%':>7} {'金额':>10} {'股数':>6}")
        print("-" * 85)

        total_invested = 0
        for a in allocations:
            dir_icon = "🟢"
            print(f"{dir_icon} {a['ticker']:<6} ${a['price']:>7.2f} {'LONG':>6} "
                  f"{a['score']:>5.0f} {a['confidence']:>7.0%} {a['rsi']:>5.0f} "
                  f"{a['adx']:>5.0f} {a['allocation_pct']:>6.1f}% ${a['allocation_usd']:>9,.0f} {a['shares']:>6}")
            total_invested += a["cost_basis"]

        cash_remaining = CAPITAL - total_invested
        print("-" * 85)
        print(f"{'合计':>8} {'':>6} {'':>6} {'':>5} {'':>8} {'':>6} {'':>6} "
              f"{100 - cash_remaining/CAPITAL*100:>6.1f}% ${total_invested:>9,.0f}")
        print(f"\n💵 剩余现金: ${cash_remaining:,.0f}")

        # Detailed reasoning
        print("\n📋 每笔仓位详解:")
        for a in allocations:
            print(f"\n  [{a['ticker']}]  ${a['allocation_usd']:,.0f} ({a['allocation_pct']:.1f}%)")
            print(f"    入场价: ${a['price']:.2f}  |  {a['shares']}股  |  置信度: {a['confidence']:.0%}")
            print(f"    看多理由:")
            for r in a["reasons"]:
                print(f"      ✅ {r}")
            if a["warnings"]:
                print(f"    风险提示:")
                for w in a["warnings"]:
                    print(f"      ⚠️  {w}")

    # Display SHORT warnings
    if shorts:
        print("\n" + "=" * 90)
        print("  ⚠️  看空信号 (不建议做多)")
        print("=" * 90)
        shorts_sorted = sorted(shorts, key=lambda s: s["confidence"], reverse=True)[:5]
        for s in shorts_sorted:
            print(f"  🔴 {s['ticker']:<6} ${s['price']:>7.2f}  置信度 {s['confidence']:.0%}  "
                  f"|  RSI {s['rsi']:.0f}  |  ADX {s['adx']:.0f}")
            for r in s["reasons"][:2]:
                print(f"     └─ {r}")

    # Summary
    print("\n" + "=" * 90)
    print("  ⚠️  免责声明: 以上为算法生成的参考信号，不构成投资建议。")
    print("  回测收益不代表未来表现，请自行判断风险。")
    print("=" * 90)


if __name__ == "__main__":
    import sys
    main()
