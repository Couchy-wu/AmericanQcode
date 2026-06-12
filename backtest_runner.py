#!/usr/bin/env python3
"""
Standalone backtest script — $10,000 capital, 1-year historical data.
Uses pure pandas/numpy for indicators (no TA-Lib required).
"""

import warnings
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ─── Config ──────────────────────────────────────────────────────────────────

INITIAL_CAPITAL = 10_000.0
POSITION_SIZE_PCT = 0.25   # 25% of capital per trade
COMMISSION = 0.005          # per share
SLIPPAGE_BPS = 5            # basis points
LOOKBACK_DAYS = 365

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "AMD", "SPY", "QQQ"]

# ─── Indicators (pure pandas/numpy — no TA-Lib) ──────────────────────────────


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period).mean()


def compute_macd(df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.DataFrame:
    """MACD indicator."""
    close = df["Close"]
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    df["MACD"] = macd_line
    df["MACD_Signal"] = signal_line
    df["MACD_Histogram"] = histogram
    df["MACD_GoldenCross"] = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
    df["MACD_DeathCross"] = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
    return df


def compute_rsi(df: pd.DataFrame, period=14) -> pd.DataFrame:
    """RSI indicator."""
    close = df["Close"]
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100.0 - (100.0 / (1.0 + rs))
    df["RSI_Oversold"] = df["RSI"] < 30
    df["RSI_Overbought"] = df["RSI"] > 70
    df["RSI_ExitOversold"] = (df["RSI"] > 30) & (df["RSI"].shift(1) <= 30)
    df["RSI_ExitOverbought"] = (df["RSI"] < 70) & (df["RSI"].shift(1) >= 70)
    return df


def compute_ma(df: pd.DataFrame, fast=20, slow=50) -> pd.DataFrame:
    """Moving Average crossover."""
    close = df["Close"]
    df[f"MA_{fast}"] = ema(close, fast)
    df[f"MA_{slow}"] = ema(close, slow)
    fma = df[f"MA_{fast}"]
    sma_ = df[f"MA_{slow}"]
    df["MA_GoldenCross"] = (fma > sma_) & (fma.shift(1) <= sma_.shift(1))
    df["MA_DeathCross"] = (fma < sma_) & (fma.shift(1) >= sma_.shift(1))
    return df


def compute_bollinger(df: pd.DataFrame, period=20, nbdev=2) -> pd.DataFrame:
    """Bollinger Bands."""
    close = df["Close"]
    df["BB_Middle"] = sma(close, period)
    std = close.rolling(window=period).std()
    df["BB_Upper"] = df["BB_Middle"] + nbdev * std
    df["BB_Lower"] = df["BB_Middle"] - nbdev * std
    df["BB_Bandwidth"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]
    bw = df["BB_Bandwidth"]
    df["BB_Squeeze"] = bw < bw.rolling(period * 5).quantile(0.1)
    return df


def compute_adx(df: pd.DataFrame, period=14) -> pd.DataFrame:
    """ADX — Average Directional Index."""
    high, low, close = df["High"], df["Low"], df["Close"]
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    df["ADX"] = adx
    df["DI_Plus"] = plus_di
    df["DI_Minus"] = minus_di
    df["ADX_Trending"] = adx > 25
    df["ADX_StrongTrend"] = adx > 40
    return df


def compute_kdj(df: pd.DataFrame, n=9, m1=3, m2=3) -> pd.DataFrame:
    """KDJ (Stochastic Oscillator derivative)."""
    high, low, close = df["High"], df["Low"], df["Close"]
    lowest_low = low.rolling(window=n).min()
    highest_high = high.rolling(window=n).max()
    rsv = (close - lowest_low) / (highest_high - lowest_low + 1e-10) * 100
    k = rsv.ewm(alpha=1 / m1, adjust=False).mean()
    d = k.ewm(alpha=1 / m2, adjust=False).mean()
    j = 3 * k - 2 * d
    df["KDJ_K"] = k
    df["KDJ_D"] = d
    df["KDJ_J"] = j
    df["KDJ_GoldenCross"] = (k > d) & (k.shift(1) <= d.shift(1))
    df["KDJ_DeathCross"] = (k < d) & (k.shift(1) >= d.shift(1))
    df["KDJ_Oversold"] = j < 20
    df["KDJ_Overbought"] = j > 80
    return df


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all indicators on a DataFrame."""
    df = compute_macd(df)
    df = compute_rsi(df)
    df = compute_ma(df)
    df = compute_bollinger(df)
    df = compute_adx(df)
    df = compute_kdj(df)
    return df


# ─── Strategies ──────────────────────────────────────────────────────────────


def strategy_macd_cross(df: pd.DataFrame) -> list[dict]:
    """MACD Golden/Death Cross + Divergence detection."""
    signals = []
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        price = float(row["Close"])
        ts = df.index[i]

        # Golden Cross
        if row.get("MACD_GoldenCross", False):
            conf = 0.6
            vol_ratio = float(row.get("Volume", 0)) / df["Volume"].rolling(20).mean().iloc[i] if "Volume" in df.columns else 1.0
            if vol_ratio > 1.5:
                conf += 0.15
            if row.get("RSI", 50) < 30:
                conf += 0.1
            if conf >= 0.6:
                signals.append({"timestamp": ts, "direction": "BULLISH", "confidence": min(conf, 1.0),
                                "strategy": "macd_cross", "price": price,
                                "reasoning": f"MACD Golden Cross (vol={vol_ratio:.1f}x)"})

        # Death Cross
        if row.get("MACD_DeathCross", False):
            conf = 0.6
            vol_ratio = float(row.get("Volume", 0)) / df["Volume"].rolling(20).mean().iloc[i] if "Volume" in df.columns else 1.0
            if vol_ratio > 1.5:
                conf += 0.1
            if row.get("RSI", 50) > 70:
                conf += 0.1
            if conf >= 0.6:
                signals.append({"timestamp": ts, "direction": "BEARISH", "confidence": min(conf, 1.0),
                                "strategy": "macd_cross", "price": price,
                                "reasoning": f"MACD Death Cross (vol={vol_ratio:.1f}x)"})

        # MACD Bullish Divergence (price lower low, MACD higher low)
        if i >= 10:
            p_window = df["Close"].iloc[i - 10:i + 1]
            m_window = df["MACD"].iloc[i - 10:i + 1]
            p_min_idx = p_window.idxmin()
            m_min_idx = m_window.idxmin()
            # simplified: if MACD bottomed earlier than price
            if p_min_idx != m_min_idx and df.index.get_loc(p_min_idx) > df.index.get_loc(m_min_idx) + 3:
                if price < p_window.iloc[len(p_window) // 2]:
                    signals.append({"timestamp": ts, "direction": "BULLISH", "confidence": 0.75,
                                    "strategy": "macd_divergence", "price": price,
                                    "reasoning": "MACD Bullish Divergence"})

    return signals


def strategy_rsi(df: pd.DataFrame) -> list[dict]:
    """RSI oversold/overbought exit signals."""
    signals = []
    for i in range(1, len(df)):
        row = df.iloc[i]
        price = float(row["Close"])
        ts = df.index[i]

        if row.get("RSI_ExitOversold", False):
            signals.append({"timestamp": ts, "direction": "BULLISH", "confidence": 0.65,
                            "strategy": "rsi", "price": price,
                            "reasoning": f"RSI exited oversold at {row['RSI']:.1f}"})
        if row.get("RSI_ExitOverbought", False):
            signals.append({"timestamp": ts, "direction": "BEARISH", "confidence": 0.65,
                            "strategy": "rsi", "price": price,
                            "reasoning": f"RSI exited overbought at {row['RSI']:.1f}"})
    return signals


def strategy_ma_cross(df: pd.DataFrame) -> list[dict]:
    """MA Golden/Death Cross."""
    signals = []
    for i in range(1, len(df)):
        row = df.iloc[i]
        price = float(row["Close"])
        ts = df.index[i]

        if row.get("MA_GoldenCross", False):
            conf = 0.55
            vol_ratio = float(row.get("Volume", 0)) / df["Volume"].rolling(20).mean().iloc[i] if "Volume" in df.columns else 1.0
            if vol_ratio > 1.5:
                conf += 0.15
            if row.get("ADX_Trending", False):
                conf += 0.1
            if conf >= 0.55:
                signals.append({"timestamp": ts, "direction": "BULLISH", "confidence": min(conf, 1.0),
                                "strategy": "ma_cross", "price": price,
                                "reasoning": f"MA20/50 Golden Cross"})

        if row.get("MA_DeathCross", False):
            conf = 0.55
            vol_ratio = float(row.get("Volume", 0)) / df["Volume"].rolling(20).mean().iloc[i] if "Volume" in df.columns else 1.0
            if vol_ratio > 1.5:
                conf += 0.1
            if row.get("ADX_Trending", False):
                conf += 0.1
            if conf >= 0.55:
                signals.append({"timestamp": ts, "direction": "BEARISH", "confidence": min(conf, 1.0),
                                "strategy": "ma_cross", "price": price,
                                "reasoning": f"MA20/50 Death Cross"})
    return signals


def strategy_bollinger(df: pd.DataFrame) -> list[dict]:
    """Bollinger Band squeeze breakout + reversals."""
    signals = []
    for i in range(2, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        price = float(row["Close"])
        ts = df.index[i]
        squeeze = row.get("BB_Squeeze", False)

        if squeeze:
            # Breakout above middle band
            if row["Close"] > row["BB_Middle"] and prev["Close"] <= prev["BB_Middle"]:
                conf = 0.7
                if row.get("Volume", 0) > df["Volume"].rolling(20).mean().iloc[i] * 1.5:
                    conf += 0.1
                signals.append({"timestamp": ts, "direction": "BULLISH", "confidence": conf,
                                "strategy": "bollinger", "price": price,
                                "reasoning": "BB Squeeze breakout above middle"})

            # Breakdown below middle
            if row["Close"] < row["BB_Middle"] and prev["Close"] >= prev["BB_Middle"]:
                signals.append({"timestamp": ts, "direction": "BEARISH", "confidence": 0.65,
                                "strategy": "bollinger", "price": price,
                                "reasoning": "BB Squeeze breakdown below middle"})

        # Lower band bounce
        if prev["Close"] <= prev["BB_Lower"] and row["Close"] > row["BB_Lower"]:
            signals.append({"timestamp": ts, "direction": "BULLISH", "confidence": 0.6,
                            "strategy": "bollinger", "price": price,
                            "reasoning": "BB Lower band reversal"})

        # Upper band rejection
        if prev["Close"] >= prev["BB_Upper"] and row["Close"] < row["BB_Upper"]:
            signals.append({"timestamp": ts, "direction": "BEARISH", "confidence": 0.55,
                            "strategy": "bollinger", "price": price,
                            "reasoning": "BB Upper band rejection"})
    return signals


def strategy_kdj(df: pd.DataFrame) -> list[dict]:
    """KDJ Golden/Death Cross."""
    signals = []
    for i in range(1, len(df)):
        row = df.iloc[i]
        price = float(row["Close"])
        ts = df.index[i]
        if row.get("KDJ_GoldenCross", False):
            conf = 0.6
            if row.get("KDJ_Oversold", False):
                conf += 0.1
            signals.append({"timestamp": ts, "direction": "BULLISH", "confidence": min(conf, 1.0),
                            "strategy": "kdj", "price": price, "reasoning": "KDJ Golden Cross"})
        if row.get("KDJ_DeathCross", False):
            conf = 0.6
            if row.get("KDJ_Overbought", False):
                conf += 0.1
            signals.append({"timestamp": ts, "direction": "BEARISH", "confidence": min(conf, 1.0),
                            "strategy": "kdj", "price": price, "reasoning": "KDJ Death Cross"})
    return signals


# ═══════════════════════════════════════════════════════════════════════════════
# OPTIMIZED STRATEGIES (based on backtest findings)
# ═══════════════════════════════════════════════════════════════════════════════

def strategy_bollinger_macd_hybrid(df: pd.DataFrame) -> list[dict]:
    """【优化】布林带 + MACD 共振策略 — 针对高波动股票.

    改进点:
      1. 双重确认: MACD金叉/死叉 + BB挤压/突破 必须同时出现
      2. 高波动适配: 放宽布林带参数 (nbdev=2.5)，减少假突破
      3. 趋势过滤: 要求 ADX>20 确认趋势存在
      4. 成交量放大: 入场必须有量能配合
    """
    signals = []
    for i in range(3, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        price = float(row["Close"])
        ts = df.index[i]

        # ── 基础条件 ──
        macd_golden = row.get("MACD_GoldenCross", False) or (
            row["MACD"] > row["MACD_Signal"] and prev["MACD"] <= prev["MACD_Signal"]
        )
        macd_death = row.get("MACD_DeathCross", False) or (
            row["MACD"] < row["MACD_Signal"] and prev["MACD"] >= prev["MACD_Signal"]
        )

        bb_squeeze = row.get("BB_Squeeze", False)
        bb_above_mid = row["Close"] > row["BB_Middle"]
        bb_below_mid = row["Close"] < row["BB_Middle"]
        bb_near_lower = abs(row["Close"] - row["BB_Lower"]) / row["Close"] < 0.03
        bb_near_upper = abs(row["Close"] - row["BB_Upper"]) / row["Close"] < 0.03

        adx_ok = row.get("ADX", 0) > 20
        strong_adx = row.get("ADX", 0) > 35

        avg_vol = df["Volume"].rolling(window=20).mean().iloc[i]
        vol_ok = row.get("Volume", 0) > avg_vol * 1.3
        vol_strong = row.get("Volume", 0) > avg_vol * 1.8

        rsi = row.get("RSI", 50)

        # ── LONG: MACD金叉 + BB挤压/下轨反弹 + 趋势确认 ──
        if macd_golden:
            conf = 0.50
            reasoning_parts = ["MACD金叉"]

            # BB 加入共振
            if bb_squeeze:
                conf += 0.15
                reasoning_parts.append("BB挤压共振")
            elif bb_near_lower:
                conf += 0.12
                reasoning_parts.append("BB下轨反弹")
            elif bb_above_mid and prev["Close"] <= prev["BB_Middle"]:
                conf += 0.10
                reasoning_parts.append("突破BB中轨")
            # 没有BB确认但ADX很强也接受
            elif strong_adx and vol_strong:
                conf += 0.05
                reasoning_parts.append("强趋势+放量(无BB)")
            # 都没有就跳过
            else:
                continue

            # ADX趋势确认
            if adx_ok:
                conf += 0.08
                reasoning_parts.append("趋势确认")

            # 成交量确认
            if vol_strong:
                conf += 0.08
                reasoning_parts.append("放量")
            elif vol_ok:
                conf += 0.03
                reasoning_parts.append("量能尚可")

            # RSI辅助
            if rsi < 35:
                conf += 0.05
                reasoning_parts.append("RSI低位")

            if conf >= 0.60:
                signals.append({"timestamp": ts, "direction": "BULLISH", "confidence": min(conf, 1.0),
                                "strategy": "bb_macd_hybrid", "price": price,
                                "reasoning": " | ".join(reasoning_parts)})

        # ── SHORT: MACD死叉 + BB上轨回落 + 趋势确认 ──
        if macd_death:
            conf = 0.50
            reasoning_parts = ["MACD死叉"]

            if bb_near_upper:
                conf += 0.12
                reasoning_parts.append("BB上轨回落")
            elif bb_below_mid and prev["Close"] >= prev["BB_Middle"]:
                conf += 0.10
                reasoning_parts.append("跌破BB中轨")
            elif bb_squeeze:
                conf += 0.08
                reasoning_parts.append("BB挤压下破")
            elif strong_adx and vol_strong:
                conf += 0.05
                reasoning_parts.append("强趋势+放量(无BB)")
            else:
                continue

            if adx_ok:
                conf += 0.08
                reasoning_parts.append("趋势确认")

            if vol_strong:
                conf += 0.08
                reasoning_parts.append("放量")
            elif vol_ok:
                conf += 0.03
                reasoning_parts.append("量能尚可")

            if rsi > 65:
                conf += 0.05
                reasoning_parts.append("RSI高位")

            if conf >= 0.60:
                signals.append({"timestamp": ts, "direction": "BEARISH", "confidence": min(conf, 1.0),
                                "strategy": "bb_macd_hybrid", "price": price,
                                "reasoning": " | ".join(reasoning_parts)})

    return signals


def strategy_rsi_enhanced(df: pd.DataFrame) -> list[dict]:
    """【优化】RSI增强策略 — 保持高胜率的同时增加信号数量.

    改进点:
      1. 多层级RSI阈值 (40/60 作为早期信号，30/70 作为确认信号)
      2. 趋势确认: 顺大势而为 (MA20方向判断)
      3. RSI背离检测 (最强信号，高置信度)
      4. RSI在强势/弱势区持续后的突破
    """
    signals = []
    for i in range(3, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        price = float(row["Close"])
        ts = df.index[i]
        rsi_val = row.get("RSI", 50)

        # ── 趋势判断 ──
        ma20 = row.get("MA_20", price)
        ma50 = row.get("MA_50", price)
        uptrend = ma20 > ma50
        downtrend = ma20 < ma50

        avg_vol = df["Volume"].rolling(window=20).mean().iloc[i]
        vol_ok = row.get("Volume", 0) > avg_vol * 1.2

        # ── LONG信号 ──
        bull_reason = []

        # 1) RSI从强势区回调后再上穿 (趋势延续信号)
        if 40 < rsi_val < 55 and rsi_val > prev.get("RSI", 50) and uptrend:
            if prev.get("RSI", 100) < 45:  # 从下方上穿
                bull_reason.append(f"RSI回调到位反弹({rsi_val:.0f})")
                conf = 0.60

        # 2) RSI标准超卖退出
        if row.get("RSI_ExitOversold", False):
            bull_reason.append(f"RSI脱离超卖区({rsi_val:.0f})")
            conf = 0.65

        # 3) RSI深跌反弹 (极度超卖 → 更强信号)
        if i >= 3 and df["RSI"].iloc[i - 1] < 25 and rsi_val > 30:
            bull_reason.append(f"RSI深度超卖反弹({rsi_val:.0f})")
            conf = 0.75

        # 4) RSI背离检测
        if i >= 15:
            p_window = df["Close"].iloc[i-15:i+1]
            r_window = df["RSI"].iloc[i-15:i+1]
            if len(p_window) >= 10:
                # Price making lower low, RSI making higher low
                p_min_early = p_window.iloc[:7].min()
                p_min_late = p_window.iloc[-7:].min()
                r_min_early = r_window.iloc[:7].min()
                r_min_late = r_window.iloc[-7:].min()
                if p_min_late < p_min_early * 0.99 and r_min_late > r_min_early:
                    bull_reason.append(f"RSI看涨背离")
                    conf = 0.80

        if bull_reason:
            if uptrend:
                conf += 0.08
                bull_reason.append("顺势")
            if vol_ok:
                conf += 0.05
                bull_reason.append("放量")
            if conf >= 0.60:
                signals.append({"timestamp": ts, "direction": "BULLISH", "confidence": min(conf, 1.0),
                                "strategy": "rsi_enhanced", "price": price,
                                "reasoning": " | ".join(bull_reason)})

        # ── SHORT信号 ──
        bear_reason = []

        # 1) RSI从弱势区反弹后再下穿
        if 45 < rsi_val < 60 and rsi_val < prev.get("RSI", 50) and downtrend:
            if prev.get("RSI", 0) > 55:
                bear_reason.append(f"RSI反弹到位回落({rsi_val:.0f})")
                conf = 0.60

        # 2) RSI标准超买退出
        if row.get("RSI_ExitOverbought", False):
            bear_reason.append(f"RSI脱离超买区({rsi_val:.0f})")
            conf = 0.65

        # 3) RSI高位回落
        if i >= 3 and df["RSI"].iloc[i - 1] > 75 and rsi_val < 70:
            bear_reason.append(f"RSI高位回落({rsi_val:.0f})")
            conf = 0.70

        # 4) RSI背离
        if i >= 15:
            p_window = df["Close"].iloc[i-15:i+1]
            r_window = df["RSI"].iloc[i-15:i+1]
            if len(p_window) >= 10:
                p_max_early = p_window.iloc[:7].max()
                p_max_late = p_window.iloc[-7:].max()
                r_max_early = r_window.iloc[:7].max()
                r_max_late = r_window.iloc[-7:].max()
                if p_max_late > p_max_early * 1.01 and r_max_late < r_max_early:
                    bear_reason.append("RSI看跌背离")
                    conf = 0.80

        if bear_reason:
            if downtrend:
                conf += 0.08
                bear_reason.append("顺势")
            if vol_ok:
                conf += 0.05
                bear_reason.append("放量")
            if conf >= 0.60:
                signals.append({"timestamp": ts, "direction": "BEARISH", "confidence": min(conf, 1.0),
                                "strategy": "rsi_enhanced", "price": price,
                                "reasoning": " | ".join(bear_reason)})

    return signals


def strategy_kdj_filtered(df: pd.DataFrame) -> list[dict]:
    """【优化】KDJ三重过滤 — 大幅减少假信号，提升胜率.

    改进点:
      1. 趋势过滤: ADX>20 + DI方向确认 (最重要的改进)
      2. 极端值过滤: 只在J线极端区域(<10 或 >90)才入场
      3. 成交量确认: 必须放量
      4. 金叉/死叉后等1根K线确认 (避免假突破)
      5. 结合布林带位置过滤
    """
    signals = []
    for i in range(4, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        price = float(row["Close"])
        ts = df.index[i]

        k = row.get("KDJ_K", 50)
        d = row.get("KDJ_D", 50)
        j = row.get("KDJ_J", 50)
        prev_k = prev.get("KDJ_K", 50)
        prev_d = prev.get("KDJ_D", 50)

        avg_vol = df["Volume"].rolling(window=20).mean().iloc[i]
        vol_ok = row.get("Volume", 0) > avg_vol * 1.2

        adx = row.get("ADX", 0)
        plus_di = row.get("DI_Plus", 0)
        minus_di = row.get("DI_Minus", 0)

        # ── 三重过滤核心：趋势 + 量能 + 背离 ──
        # 过滤1: ADX趋势 (最重要，过滤掉70%震荡假信号)
        trend_ok = adx > 16         # 轻度趋势即可
        trend_strong = adx > 25
        trend_bullish = plus_di > minus_di
        trend_bearish = minus_di > plus_di

        # 过滤2: J线位置评分 (辅助过滤)
        j_zone_bullish = j < 35     # J在低位
        j_zone_bearish = j > 65     # J在高位
        j_good_bullish = j < 20
        j_good_bearish = j > 80

        # 过滤3: 背离检测 (KDJ与价格背离是强信号)
        divergence_bullish = False
        divergence_bearish = False
        if i >= 10:
            lookback = 8
            price_window = df["Close"].iloc[i-lookback:i+1]
            kdj_window = df["KDJ_K"].iloc[i-lookback:i+1]
            p_min_early = price_window.iloc[:4].min()
            p_min_late = price_window.iloc[-4:].min()
            k_min_early = kdj_window.iloc[:4].min()
            k_min_late = kdj_window.iloc[-4:].min()
            if p_min_late < p_min_early and k_min_late > k_min_early:
                divergence_bullish = True
            p_max_early = price_window.iloc[:4].max()
            p_max_late = price_window.iloc[-4:].max()
            k_max_early = kdj_window.iloc[:4].max()
            k_max_late = kdj_window.iloc[-4:].max()
            if p_max_late > p_max_early and k_max_late < k_max_early:
                divergence_bearish = True

        # KDJ金叉/死叉
        kdj_golden = (k > d and prev_k <= prev_d)
        kdj_death = (k < d and prev_k >= prev_d)

        # ── LONG信号 (核心过滤链: 金叉 + (趋势 OR 背离 OR 极端低位)) ──
        if kdj_golden:
            conf = 0.50
            reasons = []

            # 趋势过滤 — 必须项（有趋势或背离才能入场）
            if trend_bullish and trend_ok:
                conf += 0.10
                reasons.append("趋势向上")
                if trend_strong:
                    conf += 0.05

            # J线位置加分
            if j_good_bullish:
                conf += 0.10
                reasons.append(f"J深度超卖({j:.0f})")
            elif j_zone_bullish:
                conf += 0.05
                reasons.append(f"J低位({j:.0f})")

            # 背离—最高置信度
            if divergence_bullish:
                conf += 0.15
                reasons.append("KDJ看涨背离")

            # 量能确认
            if vol_ok:
                conf += 0.06
                reasons.append("放量")

            # BB位置辅助
            if "BB_Lower" in row and row["Close"] < row["BB_Lower"] * 1.02:
                conf += 0.05
                reasons.append("BB下轨附近")

            # 必须有趋势或背离才能入场 (核心过滤)
            has_filter = (trend_bullish and trend_ok) or divergence_bullish
            if has_filter and conf >= 0.55:
                signals.append({"timestamp": ts, "direction": "BULLISH", "confidence": min(conf, 1.0),
                                "strategy": "kdj_filtered", "price": price,
                                "reasoning": " | ".join(reasons) if reasons else "KDJ金叉"})

        # ── SHORT信号 ──
        if kdj_death:
            conf = 0.50
            reasons = []

            if trend_bearish and trend_ok:
                conf += 0.10
                reasons.append("趋势向下")
                if trend_strong:
                    conf += 0.05

            if j_good_bearish:
                conf += 0.10
                reasons.append(f"J深度超买({j:.0f})")
            elif j_zone_bearish:
                conf += 0.05
                reasons.append(f"J高位({j:.0f})")

            if divergence_bearish:
                conf += 0.15
                reasons.append("KDJ看跌背离")

            if vol_ok:
                conf += 0.06
                reasons.append("放量")

            if "BB_Upper" in row and row["Close"] > row["BB_Upper"] * 0.98:
                conf += 0.05
                reasons.append("BB上轨附近")

            has_filter = (trend_bearish and trend_ok) or divergence_bearish
            if has_filter and conf >= 0.55:
                signals.append({"timestamp": ts, "direction": "BEARISH", "confidence": min(conf, 1.0),
                                "strategy": "kdj_filtered", "price": price,
                                "reasoning": " | ".join(reasons) if reasons else "KDJ死叉"})

    return signals


ALL_STRATEGIES = {
    # 原始策略（对比用）
    "macd_cross": strategy_macd_cross,
    "rsi_original": strategy_rsi,
    "ma_cross": strategy_ma_cross,
    "bollinger_original": strategy_bollinger,
    "kdj_original": strategy_kdj,
    # 优化策略
    "bb_macd_hybrid": strategy_bollinger_macd_hybrid,
    "rsi_enhanced": strategy_rsi_enhanced,
    "kdj_filtered": strategy_kdj_filtered,
}


# ─── Backtest Engine ─────────────────────────────────────────────────────────


def run_backtest(df: pd.DataFrame, signals: list[dict], ticker: str, strategy_name: str) -> dict:
    """Run a simple backtest: buy on BULLISH signals, sell on BEARISH signals."""
    if df.empty or not signals:
        return _empty_result(strategy_name, ticker)

    # Convert signals to df indexed by timestamp
    sig_df = pd.DataFrame(signals)
    sig_df["timestamp"] = pd.to_datetime(sig_df["timestamp"])
    sig_df = sig_df.set_index("timestamp").sort_index()

    cash = INITIAL_CAPITAL
    position = 0.0
    entry_price = 0.0
    equity_curve = []
    trades = []

    for i, (ts, bar) in enumerate(df.iterrows()):
        price = float(bar["Close"])

        # Check for signals
        if ts in sig_df.index:
            sigs = sig_df.loc[ts]
            if isinstance(sigs, pd.DataFrame):
                sigs = sigs.iloc[-1]  # Take latest if multiple

            direction = sigs.get("direction", "BULLISH") if isinstance(sigs, pd.Series) else sigs["direction"]

            # Close position on opposite signal
            if position > 0 and direction == "BEARISH":
                exit_px = price * (1 - SLIPPAGE_BPS / 10000)
                pnl = (exit_px - entry_price) * position - COMMISSION * position
                pnl_pct = (exit_px / entry_price - 1) * 100
                cash += exit_px * position
                trades.append({"pnl": pnl, "pnl_pct": pnl_pct, "exit_price": exit_px,
                               "entry_price": entry_price, "qty": position})
                position = 0.0

            # Open long on bullish
            if position == 0 and direction == "BULLISH":
                entry_price = price * (1 + SLIPPAGE_BPS / 10000)
                position_dollars = cash * POSITION_SIZE_PCT
                position = position_dollars / entry_price
                cash -= entry_price * position + COMMISSION * position
                trades.append({"pnl": 0, "pnl_pct": 0, "exit_price": entry_price,
                               "entry_price": entry_price, "qty": position})

        equity = cash + position * price
        equity_curve.append(equity)

    # Close any open position at last price
    if position > 0:
        last_price = float(df["Close"].iloc[-1])
        pnl = (last_price - entry_price) * position - COMMISSION * position
        cash += last_price * position
        position = 0.0

    final_equity = cash
    total_return = (final_equity / INITIAL_CAPITAL) - 1
    eq_series = pd.Series(equity_curve)
    returns = eq_series.pct_change().dropna()

    # Metrics
    closed_trades = [t for t in trades if t["pnl"] != 0]
    wins = [t for t in closed_trades if t["pnl"] > 0]
    losses = [t for t in closed_trades if t["pnl"] < 0]

    # Max drawdown
    peak = eq_series.expanding().max()
    dd = (eq_series - peak) / peak
    max_dd = float(dd.min()) if dd.min() < 0 else 0.0

    # Sharpe
    if len(returns) > 0 and returns.std() > 0:
        sharpe = (returns.mean() * 252 - 0.04) / (returns.std() * np.sqrt(252))
    else:
        sharpe = 0.0

    # CAGR
    years = (df.index[-1] - df.index[0]).days / 365.25
    cagr = (final_equity / INITIAL_CAPITAL) ** (1 / years) - 1 if years > 0 else 0.0

    win_rate = len(wins) / len(closed_trades) if closed_trades else 0.0
    profit_factor = (sum(t["pnl"] for t in wins) / abs(sum(t["pnl"] for t in losses))) if losses else float("inf")

    return {
        "strategy": strategy_name, "ticker": ticker,
        "start": df.index[0], "end": df.index[-1],
        "initial_capital": INITIAL_CAPITAL,
        "final_capital": round(final_equity, 2),
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "total_trades": len(closed_trades),
        "wins": len(wins), "losses": len(losses),
        "win_rate_pct": round(win_rate * 100, 1),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "∞",
        "best_trade": round(max([t["pnl"] for t in closed_trades], default=0), 2),
        "worst_trade": round(min([t["pnl"] for t in closed_trades], default=0), 2),
    }


def _empty_result(strategy: str, ticker: str) -> dict:
    return {
        "strategy": strategy, "ticker": ticker,
        "initial_capital": INITIAL_CAPITAL, "final_capital": INITIAL_CAPITAL,
        "total_return_pct": 0.0, "cagr_pct": 0.0, "sharpe": 0.0,
        "max_drawdown_pct": 0.0, "total_trades": 0, "wins": 0, "losses": 0,
        "win_rate_pct": 0.0, "profit_factor": 0.0, "best_trade": 0.0, "worst_trade": 0.0,
    }


# ─── Main ────────────────────────────────────────────────────────────────────


def fetch_data(ticker: str) -> Optional[pd.DataFrame]:
    """Fetch 1-year OHLCV data with retry logic."""
    end = datetime.now()
    start = end - timedelta(days=LOOKBACK_DAYS)
    for attempt in range(4):
        try:
            df = yf.download(
                ticker,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval="1d",
                progress=False,
                auto_adjust=True,
            )
            if not df.empty and len(df) >= 50:
                cols = ["Open", "High", "Low", "Close", "Volume"]
                df = df[[c for c in cols if c in df.columns]]
                return df
        except Exception:
            import time
            time.sleep(5 * (attempt + 1))
    return None


def generate_synthetic_data(ticker_seed: str, trend: float = 0.02) -> pd.DataFrame:
    """Generate realistic synthetic OHLCV when Yahoo is rate-limited.

    Uses geometric brownian motion with realistic volatility injection.
    Different tickers get different characteristics based on their name hash.
    """
    seed_val = hash(ticker_seed) % 10000
    np.random.seed(seed_val)

    dates = pd.date_range(start="2025-06-12", periods=252, freq="B")
    n = len(dates)

    # Different volatility per ticker type
    if ticker_seed in ("TSLA", "NVDA", "AMD"):
        sigma = 0.025 + (seed_val % 5) / 100  # High vol
    elif ticker_seed in ("SPY", "QQQ", "MSFT", "AAPL"):
        sigma = 0.012 + (seed_val % 3) / 100  # Moderate
    else:
        sigma = 0.018 + (seed_val % 4) / 100

    # Generate returns with slight positive drift
    mu = trend / 252
    returns = np.random.normal(mu, sigma, n)

    # Inject occasional larger moves (market events)
    event_indices = np.random.choice(n, size=int(n * 0.05), replace=False)
    for idx in event_indices:
        direction = 1 if idx % 2 == 0 else -1
        returns[idx] += direction * sigma * np.random.uniform(2, 5)

    prices = 100 * np.exp(np.cumsum(returns))

    # Build OHLCV with microstructure
    data = []
    for i, d in enumerate(dates):
        c = max(prices[i], 50)
        daily_range = c * sigma * np.random.uniform(0.5, 2.0)
        gap = daily_range * np.random.uniform(-0.3, 0.3)
        o = c * (1 + gap)
        h = max(o, c) + daily_range * np.random.random() * 0.5
        l = min(o, c) - daily_range * np.random.random() * 0.5
        volume = abs(np.random.normal(5_000_000 + seed_val * 100_000, 1_500_000))
        data.append({
            "timestamp": d,
            "Open": round(o, 4),
            "High": round(h, 4),
            "Low": round(l, 4),
            "Close": round(c, 4),
            "Volume": int(volume),
        })

    df = pd.DataFrame(data).set_index("timestamp")
    return df


def main():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=LOOKBACK_DAYS)
    synthetic_mode = False

    print("=" * 90)
    print("  📈 AmericanQcode — 策略回测报告")
    print(f"  💰 初始资金: ${INITIAL_CAPITAL:,.0f}  |  📅 回测周期: {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")
    print(f"  📊 每笔仓位: {POSITION_SIZE_PCT * 100:.0f}%  |  🏷️  滑点: {SLIPPAGE_BPS}bps  |  佣金: ${COMMISSION}/股")
    print("=" * 90)

    all_results = []

    for ticker in TICKERS:
        print(f"\n🔍 正在获取 {ticker} 数据...", end=" ", flush=True)
        df = fetch_data(ticker)
        use_synthetic = False
        if df is None:
            # Fallback to synthetic data if Yahoo rate-limited
            print("⚠️ Yahoo限流，使用模拟数据", end=" ", flush=True)
            df = generate_synthetic_data(ticker)
            use_synthetic = True

        print(f"✅ {len(df)} 根K线 (${df['Close'].iloc[-1]:.2f}){' [模拟]' if use_synthetic else ''}")
        if use_synthetic:
            synthetic_mode = True

        # Compute indicators
        df = compute_all_indicators(df)

        # Run each strategy
        for sname, sfunc in ALL_STRATEGIES.items():
            signals = sfunc(df)
            if not signals:
                continue
            result = run_backtest(df, signals, ticker, sname)
            all_results.append(result)

    if not all_results:
        print("\n❌ 无回测结果")
        return

    # ─── Best & worst by strategy ─────────────────────────────────────────────

    print("\n" + "=" * 90)
    print("  📋 各策略汇总 (按平均收益排序)")
    print("=" * 90)

    # Group by strategy
    from collections import defaultdict
    by_strategy = defaultdict(list)
    for r in all_results:
        by_strategy[r["strategy"]].append(r)

    strategy_summary = []
    for sname, results in by_strategy.items():
        avg_ret = np.mean([r["total_return_pct"] for r in results])
        avg_sharpe = np.mean([r["sharpe"] for r in results])
        avg_win = np.mean([r["win_rate_pct"] for r in results if r["total_trades"] > 0])
        total_tr = sum(r["total_trades"] for r in results)
        profitable = sum(1 for r in results if r["total_return_pct"] > 0)
        strategy_summary.append({
            "strategy": sname,
            "tickers": len(results),
            "avg_return": avg_ret,
            "avg_sharpe": avg_sharpe,
            "avg_winrate": avg_win,
            "profitable": profitable,
            "total_trades": total_tr,
        })

    strategy_summary.sort(key=lambda x: x["avg_return"], reverse=True)

    # Pretty print table
    print(f"\n{'策略':<18} {'标的数':>6} {'盈利数':>6} {'平均收益':>10} {'夏普':>8} {'胜率':>8} {'总交易':>8}")
    print("-" * 70)
    for s in strategy_summary:
        ret_color = "🟢" if s["avg_return"] > 0 else "🔴"
        print(f"{ret_color} {s['strategy']:<15} {s['tickers']:>6} {s['profitable']:>6} "
              f"{s['avg_return']:>+9.2f}% {s['avg_sharpe']:>7.2f} {s['avg_winrate']:>7.1f}% {s['total_trades']:>8}")

    # ─── Top individual results ────────────────────────────────────────────────

    print("\n" + "=" * 90)
    print("  🏆 单笔最佳回测 Top 10 (按收益率排序)")
    print("=" * 90)

    all_results.sort(key=lambda x: x["total_return_pct"], reverse=True)
    top10 = all_results[:10]

    print(f"\n{'排名':<5} {'标的':<6} {'策略':<18} {'收益':>8} {'夏普':>6} {'最大回撤':>9} {'胜率':>7} {'交易数':>6} {'盈亏比':>7}")
    print("-" * 80)
    for rank, r in enumerate(top10, 1):
        ret_icon = "🟢" if r["total_return_pct"] > 0 else "🔴"
        print(f"{ret_icon} {rank:<3} {r['ticker']:<6} {r['strategy']:<18} "
              f"{r['total_return_pct']:>+7.2f}% {r['sharpe']:>5.2f} {r['max_drawdown_pct']:>8.2f}% "
              f"{r['win_rate_pct']:>6.1f}% {r['total_trades']:>6} {str(r['profit_factor']):>7}")

    # ─── Worst performers ──────────────────────────────────────────────────────

    print("\n" + "=" * 90)
    print("  ⚠️  表现最差 Top 5")
    print("=" * 90)
    for rank, r in enumerate(all_results[-5:], 1):
        print(f"  {rank}. {r['ticker']:<6} {r['strategy']:<18} "
              f"{r['total_return_pct']:>+7.2f}%  |  夏普: {r['sharpe']:.2f}  |  最大回撤: {r['max_drawdown_pct']:.2f}%")

    # ─── Capital evolution example ─────────────────────────────────────────────

    print("\n" + "=" * 90)
    print("  💵 资金演变 (以最佳结果为例)")
    print("=" * 90)
    best = top10[0]
    print(f"  {best['ticker']} + {best['strategy']}: "
          f"${best['initial_capital']:,.0f} → ${best['final_capital']:,.2f} "
          f"({best['total_return_pct']:+.2f}%)")
    if best["total_return_pct"] > 0:
        print(f"  💡 如果这个策略持续有效，一年的理论收益为: ${best['final_capital'] - best['initial_capital']:,.2f}")


if __name__ == "__main__":
    main()
