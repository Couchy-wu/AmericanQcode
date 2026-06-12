"""Tests for Moving Average indicators."""

from src.indicators.moving_averages import compute_ma_cross, compute_sma, compute_ema
from src.indicators.base import golden_cross, death_cross
import pandas as pd


def test_compute_sma(sample_ohlcv_df):
    """Test SMA computation."""
    result = compute_sma(sample_ohlcv_df, period=20)
    assert "SMA_20" in result.columns
    assert not result["SMA_20"].isna().all()


def test_compute_ema(sample_ohlcv_df):
    """Test EMA computation."""
    result = compute_ema(sample_ohlcv_df, period=20)
    assert "EMA_20" in result.columns


def test_compute_ma_cross(sample_ohlcv_df):
    """Test MA cross detection."""
    result = compute_ma_cross(sample_ohlcv_df, fast=5, slow=20)
    assert "MA_5" in result.columns
    assert "MA_20" in result.columns
    assert "MA_GoldenCross" in result.columns
    assert "MA_DeathCross" in result.columns


def test_golden_cross_basic():
    """Test golden cross detection with known data."""
    a = pd.Series([1.0, 2.0, 3.0, 4.0])
    b = pd.Series([2.0, 2.0, 2.0, 2.0])
    cross = golden_cross(a, b)
    # a crosses above b at index 2 (value 3 > 2)
    assert cross.iloc[2]
    assert not cross.iloc[0]
    assert not cross.iloc[1]


def test_death_cross_basic():
    """Test death cross detection with known data."""
    a = pd.Series([3.0, 2.0, 1.0, 1.0])
    b = pd.Series([2.0, 2.0, 2.0, 2.0])
    cross = death_cross(a, b)
    # a crosses below b at index 2 (value 1 < 2)
    assert cross.iloc[2]
    assert not cross.iloc[0]
