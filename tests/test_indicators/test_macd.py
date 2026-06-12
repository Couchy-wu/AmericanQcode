"""Tests for MACD indicator."""

import numpy as np
import pandas as pd

from src.indicators.macd import compute_macd


def test_compute_macd_basic(sample_ohlcv_df):
    """Test MACD computation returns expected columns."""
    df = sample_ohlcv_df
    result = compute_macd(df)

    assert "MACD" in result.columns
    assert "MACD_Signal" in result.columns
    assert "MACD_Histogram" in result.columns
    assert "MACD_GoldenCross" in result.columns
    assert "MACD_DeathCross" in result.columns
    assert len(result) == len(df)
    assert not result["MACD"].isna().all()


def test_compute_macd_cross_detection(sample_ohlcv_df):
    """Test that cross detection returns boolean values."""
    df = sample_ohlcv_df
    result = compute_macd(df)

    # Should have boolean crosses
    assert result["MACD_GoldenCross"].dtype == bool
    assert result["MACD_DeathCross"].dtype == bool


def test_compute_macd_with_custom_params(sample_ohlcv_df):
    """Test MACD with non-default parameters."""
    df = sample_ohlcv_df
    result = compute_macd(df, fast=10, slow=30, signal=8)
    assert len(result) == len(df)
