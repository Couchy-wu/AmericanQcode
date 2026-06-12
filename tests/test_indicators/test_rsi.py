"""Tests for RSI indicator."""

import numpy as np

from src.indicators.rsi import compute_rsi


def test_compute_rsi_basic(sample_ohlcv_df):
    """Test RSI computation returns expected columns."""
    df = sample_ohlcv_df
    result = compute_rsi(df)

    assert "RSI" in result.columns
    assert "RSI_Oversold" in result.columns
    assert "RSI_Overbought" in result.columns
    assert len(result) == len(df)


def test_rsi_range(sample_ohlcv_df):
    """Test RSI values are within 0-100 range."""
    df = sample_ohlcv_df
    result = compute_rsi(df)

    rsi_values = result["RSI"].dropna()
    assert (rsi_values >= 0).all()
    assert (rsi_values <= 100).all()
