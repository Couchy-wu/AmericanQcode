"""Shared test fixtures."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta


@pytest.fixture
def sample_ohlcv_df():
    """Generate a synthetic OHLCV DataFrame with 200 days of data."""
    dates = pd.date_range(start="2024-01-01", periods=200, freq="B")
    np.random.seed(42)

    close = 100.0
    data = []
    for i, d in enumerate(dates):
        change = np.random.normal(0, 1.5)
        close = close * (1 + change / 100)
        high = close * (1 + abs(np.random.normal(0, 0.01)))
        low = close * (1 - abs(np.random.normal(0, 0.01)))
        open_p = low + np.random.random() * (high - low)
        volume = abs(np.random.normal(1_000_000, 200_000))
        data.append({
            "timestamp": d,
            "Open": open_p,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        })

    df = pd.DataFrame(data)
    df = df.set_index("timestamp")
    return df


@pytest.fixture
def trending_up_df():
    """Generate a DataFrame with a clear uptrend."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="B")
    close = np.linspace(90, 110, 100) + np.random.normal(0, 0.3, 100)
    data = []
    for i, d in enumerate(dates):
        c = close[i]
        h = c * 1.01
        l = c * 0.99
        o = l + np.random.random() * (h - l)
        data.append({
            "timestamp": d,
            "Open": o,
            "High": h,
            "Low": l,
            "Close": c,
            "Volume": 1_000_000,
        })
    df = pd.DataFrame(data).set_index("timestamp")
    return df
