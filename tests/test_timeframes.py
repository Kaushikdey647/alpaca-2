"""Tests for :mod:`shunya.data.timeframes`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from alpaca.data.timeframe import TimeFrameUnit

from shunya.data.timeframes import (
    BarIndexPolicy,
    BarSpec,
    BarUnit,
    build_trading_calendar,
    bar_spec_is_intraday,
    bar_spec_to_alpaca_timeframe,
    bar_spec_to_yfinance_interval,
    default_bar_spec,
    normalize_bar_timestamp,
    normalize_history_index,
    resample_ohlcv_yearly,
    timestamp_is_on_trading_grid,
    trading_time_distance,
)


def test_bar_spec_validates_positive_step() -> None:
    with pytest.raises(ValueError, match="step"):
        BarSpec(BarUnit.DAYS, 0)


def test_default_bar_spec_is_daily() -> None:
    assert default_bar_spec() == BarSpec(BarUnit.DAYS, 1)


def test_bar_spec_is_intraday_units() -> None:
    assert bar_spec_is_intraday(BarSpec(BarUnit.SECONDS, 1))
    assert bar_spec_is_intraday(BarSpec(BarUnit.MINUTES, 15))
    assert bar_spec_is_intraday(BarSpec(BarUnit.HOURS, 1))
    assert not bar_spec_is_intraday(BarSpec(BarUnit.DAYS, 1))
    assert not bar_spec_is_intraday(BarSpec(BarUnit.MONTHS, 1))


def test_yfinance_interval_mapping() -> None:
    assert bar_spec_to_yfinance_interval(BarSpec(BarUnit.DAYS, 1)) == "1d"
    assert bar_spec_to_yfinance_interval(BarSpec(BarUnit.DAYS, 5)) == "5d"
    assert bar_spec_to_yfinance_interval(BarSpec(BarUnit.MINUTES, 5)) == "5m"
    assert bar_spec_to_yfinance_interval(BarSpec(BarUnit.HOURS, 1)) == "1h"
    assert bar_spec_to_yfinance_interval(BarSpec(BarUnit.WEEKS, 1)) == "1wk"
    assert bar_spec_to_yfinance_interval(BarSpec(BarUnit.MONTHS, 1)) == "1mo"
    assert (
        bar_spec_to_yfinance_interval(BarSpec(BarUnit.YEARS, 1))
        == "__monthly_then_year_resample"
    )


def test_yfinance_interval_rejects_unsupported() -> None:
    with pytest.raises(ValueError, match="sub-minute"):
        bar_spec_to_yfinance_interval(BarSpec(BarUnit.SECONDS, 1))
    with pytest.raises(ValueError, match="daily interval"):
        bar_spec_to_yfinance_interval(BarSpec(BarUnit.DAYS, 2))
    with pytest.raises(ValueError, match="minute interval"):
        bar_spec_to_yfinance_interval(BarSpec(BarUnit.MINUTES, 7))


def test_alpaca_timeframe_mapping() -> None:
    d1 = bar_spec_to_alpaca_timeframe(BarSpec(BarUnit.DAYS, 1))
    assert d1.amount == 1
    assert d1.unit == TimeFrameUnit.Day
    m5 = bar_spec_to_alpaca_timeframe(BarSpec(BarUnit.MINUTES, 5))
    assert m5.amount == 5
    assert m5.unit == TimeFrameUnit.Minute
    assert (
        bar_spec_to_alpaca_timeframe(BarSpec(BarUnit.YEARS, 1))
        == "__monthly_then_year_resample"
    )
    with pytest.raises(ValueError, match="sub-minute"):
        bar_spec_to_alpaca_timeframe(BarSpec(BarUnit.SECONDS, 1))


def test_normalize_bar_timestamp() -> None:
    ts = pd.Timestamp("2024-06-15 15:30:00")
    assert normalize_bar_timestamp(ts, BarSpec(BarUnit.MINUTES, 5)) == ts
    assert normalize_bar_timestamp(ts, BarSpec(BarUnit.DAYS, 1)) == pd.Timestamp(
        "2024-06-15"
    )


def test_normalize_history_index_daily_vs_intraday() -> None:
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-02 16:00:00+00:00")])
    df = pd.DataFrame(
        {
            "Open": [1.0],
            "High": [1.0],
            "Low": [1.0],
            "Close": [1.0],
            "Volume": [10.0],
        },
        index=idx,
    )
    daily = normalize_history_index(df, BarSpec(BarUnit.DAYS, 1))
    assert daily.index[0] == pd.Timestamp("2024-01-02")
    intra = normalize_history_index(df, BarSpec(BarUnit.MINUTES, 30))
    # Default policy: America/New_York — 16:00 UTC == 11:00 Eastern (Jan — EST).
    assert intra.index[0] == pd.Timestamp("2024-01-02 11:00:00")


def test_normalize_history_index_intraday_legacy_utc_naive() -> None:
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-02 16:00:00+00:00")])
    df = pd.DataFrame(
        {
            "Open": [1.0],
            "High": [1.0],
            "Low": [1.0],
            "Close": [1.0],
            "Volume": [10.0],
        },
        index=idx,
    )
    legacy = BarIndexPolicy(timezone="UTC")
    intra = normalize_history_index(
        df, BarSpec(BarUnit.MINUTES, 30), policy=legacy
    )
    assert intra.index[0] == pd.Timestamp("2024-01-02 16:00:00")


def test_normalize_history_index_intraday_dst_stable_ny_open() -> None:
    """US RTH open is 09:30 America/New_York before and after spring forward."""
    winter = pd.Timestamp("2026-03-02 14:30:00+00:00")
    spring = pd.Timestamp("2026-03-09 13:30:00+00:00")
    df = pd.DataFrame(
        {"Open": [1.0, 2.0], "High": [1.0, 2.0], "Low": [1.0, 2.0], "Close": [1.0, 2.0], "Volume": [1.0, 2.0]},
        index=pd.DatetimeIndex([winter, spring]),
    )
    out = normalize_history_index(df, BarSpec(BarUnit.MINUTES, 5))
    assert out.index[0] == pd.Timestamp("2026-03-02 09:30:00")
    assert out.index[1] == pd.Timestamp("2026-03-09 09:30:00")


def test_normalize_history_index_daily_anchor_utc_matches_prior_semantics() -> None:
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-02 16:00:00+00:00")])
    df = pd.DataFrame(
        {
            "Open": [1.0],
            "High": [1.0],
            "Low": [1.0],
            "Close": [1.0],
            "Volume": [10.0],
        },
        index=idx,
    )
    pol = BarIndexPolicy(timezone="UTC", daily_anchor="utc")
    daily = normalize_history_index(df, BarSpec(BarUnit.DAYS, 1), policy=pol)
    assert daily.index[0] == pd.Timestamp("2024-01-02")


def test_resample_ohlcv_yearly_aggregates_monthly() -> None:
    idx = pd.date_range("2020-01-31", "2021-12-31", freq="ME")
    n = len(idx)
    df = pd.DataFrame(
        {
            "Open": np.arange(n, dtype=float),
            "High": np.arange(n, dtype=float),
            "Low": np.arange(n, dtype=float),
            "Close": np.arange(n, dtype=float),
            "Volume": np.full(n, 10.0),
        },
        index=idx,
    )
    y = resample_ohlcv_yearly(df)
    assert len(y) == 2
    assert y.index[0].year == 2020
    assert y.index[-1].year == 2021


def test_build_trading_calendar_intraday_skips_weekends() -> None:
    cal = build_trading_calendar(
        "2024-01-05",  # Friday
        "2024-01-08",  # Monday
        BarSpec(BarUnit.HOURS, 1),
        policy=BarIndexPolicy(),
    )
    assert len(cal) > 0
    assert pd.Timestamp("2024-01-06 09:30:00") not in cal
    assert pd.Timestamp("2024-01-07 09:30:00") not in cal
    assert pd.Timestamp("2024-01-08 09:30:00") in cal


def test_timestamp_is_on_trading_grid_intraday() -> None:
    spec = BarSpec(BarUnit.MINUTES, 5)
    pol = BarIndexPolicy()
    assert timestamp_is_on_trading_grid("2024-01-02 09:30:00", spec, policy=pol)
    assert timestamp_is_on_trading_grid("2024-01-02 15:55:00", spec, policy=pol)
    assert not timestamp_is_on_trading_grid("2024-01-02 09:32:00", spec, policy=pol)
    assert not timestamp_is_on_trading_grid("2024-01-06 09:30:00", spec, policy=pol)


def test_trading_time_distance_ignores_weekend_wall_clock_gap() -> None:
    spec = BarSpec(BarUnit.MINUTES, 30)
    bars, seconds = trading_time_distance(
        "2024-01-05 15:30:00",  # Friday last 30m bar
        "2024-01-08 09:30:00",  # Monday first 30m bar
        spec,
        policy=BarIndexPolicy(),
    )
    assert bars == 1
    assert seconds == pytest.approx(30 * 60)
