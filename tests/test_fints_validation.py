"""Tests for strict OHLCV validation and feature history contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shunya.data.fints import finTs
from shunya.data.timeframes import BarIndexPolicy, BarSpec, BarUnit
from shunya.data.validation import validate_core_ohlcv_coverage


class _StubOneTicker:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def download(self, ticker_list, start, end, *, bar_spec=None, bar_index_policy=None):
        del ticker_list, start, end, bar_spec, bar_index_policy
        return self._df


class _StubMulti:
    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self._frames = frames

    def download(self, ticker_list, start, end, *, bar_spec=None, bar_index_policy=None):
        del ticker_list, start, end, bar_spec, bar_index_policy
        return pd.concat(self._frames, axis=1)


def test_validate_raises_on_missing_ticker_in_provider() -> None:
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")])
    raw = pd.concat(
        {
            "AAPL": pd.DataFrame(
                {
                    "Open": [1.0, 1.0],
                    "High": [1.0, 1.0],
                    "Low": [1.0, 1.0],
                    "Close": [1.0, 1.0],
                    "Volume": [1.0, 1.0],
                },
                index=idx,
            )
        },
        axis=1,
    )
    with pytest.raises(ValueError, match="strict_provider_universe"):
        validate_core_ohlcv_coverage(
            raw,
            ticker_list=["AAPL", "MSFT"],
            start="2024-01-01",
            end="2024-01-10",
            bar_spec=BarSpec(BarUnit.DAYS, 1),
            strict_provider_universe=True,
            strict_ohlcv=True,
            strict_empty=True,
        )


def test_validate_raises_on_nan_close() -> None:
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-02")])
    raw = pd.DataFrame(
        {
            "Open": [1.0],
            "High": [1.0],
            "Low": [1.0],
            "Close": [float("nan")],
            "Volume": [100.0],
        },
        index=idx,
    )
    with pytest.raises(ValueError, match="strict_ohlcv"):
        validate_core_ohlcv_coverage(
            raw,
            ticker_list=["AAPL"],
            start="2024-01-01",
            end="2024-01-10",
            bar_spec=BarSpec(BarUnit.DAYS, 1),
            strict_provider_universe=True,
            strict_ohlcv=True,
            strict_empty=True,
        )


def test_validate_intraday_accepts_ny_naive_within_requested_window() -> None:
    raw = pd.DataFrame(
        {
            "Open": [1.0],
            "High": [1.0],
            "Low": [1.0],
            "Close": [1.0],
            "Volume": [1.0],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2024-01-02 09:30:00")]),
    )
    validate_core_ohlcv_coverage(
        raw,
        ticker_list=["AAPL"],
        start="2024-01-02",
        end="2024-01-02",
        bar_spec=BarSpec(BarUnit.MINUTES, 5),
        strict_provider_universe=True,
        strict_ohlcv=True,
        strict_empty=True,
        bar_index_policy=BarIndexPolicy(),
    )


def test_validate_intraday_rejects_when_bar_before_start_date_ny_window() -> None:
    raw = pd.DataFrame(
        {
            "Open": [1.0],
            "High": [1.0],
            "Low": [1.0],
            "Close": [1.0],
            "Volume": [1.0],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2024-01-01 09:30:00")]),
    )
    with pytest.raises(ValueError, match="provider_index_out_of_range"):
        validate_core_ohlcv_coverage(
            raw,
            ticker_list=["AAPL"],
            start="2024-01-02",
            end="2024-01-02",
            bar_spec=BarSpec(BarUnit.MINUTES, 5),
            strict_provider_universe=True,
            strict_ohlcv=True,
            strict_empty=True,
            bar_index_policy=BarIndexPolicy(),
        )


def test_validate_intraday_utc_policy_accepts_utc_wall_clock() -> None:
    raw = pd.DataFrame(
        {
            "Open": [1.0],
            "High": [1.0],
            "Low": [1.0],
            "Close": [1.0],
            "Volume": [1.0],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2024-01-02 14:30:00")]),
    )
    validate_core_ohlcv_coverage(
        raw,
        ticker_list=["AAPL"],
        start="2024-01-02",
        end="2024-01-02",
        bar_spec=BarSpec(BarUnit.MINUTES, 5),
        strict_provider_universe=True,
        strict_ohlcv=True,
        strict_empty=True,
        bar_index_policy=BarIndexPolicy(timezone="UTC"),
    )


def test_validate_intraday_strict_grid_rejects_off_grid_timestamp() -> None:
    raw = pd.DataFrame(
        {
            "Open": [1.0],
            "High": [1.0],
            "Low": [1.0],
            "Close": [1.0],
            "Volume": [1.0],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2024-01-02 09:32:00")]),
    )
    with pytest.raises(ValueError, match="strict_trading_grid"):
        validate_core_ohlcv_coverage(
            raw,
            ticker_list=["AAPL"],
            start="2024-01-02",
            end="2024-01-02",
            bar_spec=BarSpec(BarUnit.MINUTES, 5),
            strict_provider_universe=True,
            strict_ohlcv=True,
            strict_empty=True,
            bar_index_policy=BarIndexPolicy(),
            strict_trading_grid=True,
        )


def test_finTs_insufficient_history_full_features() -> None:
    idx = pd.date_range("2024-01-02", periods=50, freq="B")
    raw = pd.concat(
        {
            "AAPL": pd.DataFrame(
                {
                    "Open": np.ones(len(idx)),
                    "High": np.ones(len(idx)),
                    "Low": np.ones(len(idx)),
                    "Close": np.ones(len(idx)),
                    "Volume": np.ones(len(idx)) * 1e6,
                },
                index=idx,
            ),
            "MSFT": pd.DataFrame(
                {
                    "Open": np.ones(len(idx)),
                    "High": np.ones(len(idx)),
                    "Low": np.ones(len(idx)),
                    "Close": np.ones(len(idx)),
                    "Volume": np.ones(len(idx)) * 1e6,
                },
                index=idx,
            ),
        },
        axis=1,
    )
    with pytest.raises(ValueError, match="insufficient_bars_for_features"):
        finTs(
            "2024-01-01",
            "2024-12-31",
            ["AAPL", "MSFT"],
            market_data=_StubMulti(
                {
                    "AAPL": raw["AAPL"],
                    "MSFT": raw["MSFT"],
                }
            ),
            attach_yfinance_classifications=False,
            feature_mode="full",
        )


def test_align_report_includes_n_bars() -> None:
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")])
    raw = pd.concat(
        {
            "AAPL": pd.DataFrame(
                {
                    "Open": [1.0, 1.0],
                    "High": [1.0, 1.0],
                    "Low": [1.0, 1.0],
                    "Close": [1.0, 1.0],
                    "Volume": [1.0, 1.0],
                },
                index=idx,
            ),
            "MSFT": pd.DataFrame(
                {
                    "Open": [1.0, 1.0],
                    "High": [1.0, 1.0],
                    "Low": [1.0, 1.0],
                    "Close": [1.0, 1.0],
                    "Volume": [1.0, 1.0],
                },
                index=idx,
            ),
        },
        axis=1,
    )
    fts = finTs(
        "2024-01-01",
        "2024-01-10",
        ["AAPL", "MSFT"],
        market_data=_StubMulti({"AAPL": raw["AAPL"], "MSFT": raw["MSFT"]}),
        attach_yfinance_classifications=False,
        feature_mode="ohlcv_only",
        require_history_bars=1,
    )
    rep = fts.align_universe(
        ("Open", "High", "Low", "Close", "Volume", "VWAP"),
        on_bad_ticker="drop",
    )
    d = rep.as_dict()
    assert d["n_bars"] == d["n_days"] == 2


def test_strict_empty_raises() -> None:
    class _Empty:
        def download(self, *args, **kwargs):
            return pd.DataFrame()

    with pytest.raises(ValueError, match="strict_empty"):
        finTs(
            "2024-01-01",
            "2024-01-10",
            ["AAPL"],
            market_data=_Empty(),
            attach_yfinance_classifications=False,
            feature_mode="ohlcv_only",
        )
