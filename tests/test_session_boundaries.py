"""Intraday session isolation for signal lag, decay reset wiring, and concat plots."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shunya.algorithm.finstrat import FinStrat
from shunya.data.fints import finTs
from shunya.data.timeframes import BarSpec, BarUnit, default_bar_index_policy


def _intraday_two_session_panel() -> finTs:
    """Shared 1h bar calendar: two bars on day A, two on day B (naive NY wall)."""
    d1 = pd.Timestamp("2024-01-02 10:00:00")
    d2 = pd.Timestamp("2024-01-02 11:00:00")
    d3 = pd.Timestamp("2024-01-03 10:00:00")
    d4 = pd.Timestamp("2024-01-03 11:00:00")
    tickers = ["AAA", "BBB"]
    cal = [d1, d2, d3, d4]
    rows = [(t, d) for t in tickers for d in cal]
    idx = pd.MultiIndex.from_tuples(rows, names=["Ticker", "Date"])
    n = len(rows)
    p = 100.0
    df = pd.DataFrame(
        {
            "Open": [p] * n,
            "High": [p + 1.0] * n,
            "Low": [p - 1.0] * n,
            "Close": [p] * n,
            "Volume": [1e6] * n,
        },
        index=idx,
    )
    stub = object.__new__(finTs)
    stub.start_date = cal[0]
    stub.end_date = cal[-1]
    stub.session = None
    stub.ticker_list = list(tickers)
    stub.df = df
    stub._aligned_calendar = None
    stub.bar_spec = BarSpec(BarUnit.HOURS, 1)
    stub._bar_index_policy = default_bar_index_policy()
    stub._trading_axis_mode = "observed"
    return stub


def _intraday_two_session_panel_minutes() -> finTs:
    d1 = pd.Timestamp("2024-01-02 09:30:00")
    d2 = pd.Timestamp("2024-01-02 09:35:00")
    d3 = pd.Timestamp("2024-01-03 09:30:00")
    d4 = pd.Timestamp("2024-01-03 09:35:00")
    tickers = ["AAA", "BBB"]
    cal = [d1, d2, d3, d4]
    rows = [(t, d) for t in tickers for d in cal]
    idx = pd.MultiIndex.from_tuples(rows, names=["Ticker", "Date"])
    n = len(rows)
    p = 100.0
    df = pd.DataFrame(
        {
            "Open": [p] * n,
            "High": [p + 1.0] * n,
            "Low": [p - 1.0] * n,
            "Close": [p] * n,
            "Volume": [1e6] * n,
        },
        index=idx,
    )
    stub = object.__new__(finTs)
    stub.start_date = cal[0]
    stub.end_date = cal[-1]
    stub.session = None
    stub.ticker_list = list(tickers)
    stub.df = df
    stub._aligned_calendar = None
    stub.bar_spec = BarSpec(BarUnit.MINUTES, 5)
    stub._bar_index_policy = default_bar_index_policy()
    stub._trading_axis_mode = "observed"
    return stub


def test_execution_lag_forbid_cross_session_first_bar_raises() -> None:
    fts = _intraday_two_session_panel()
    with pytest.raises(ValueError, match="cross_session_signal_lag"):
        fts.execution_lag_calendar_date(
            pd.Timestamp("2024-01-03 10:00:00"),
            lag=1,
            forbid_cross_session=True,
        )


def test_execution_lag_forbid_cross_session_second_bar_ok() -> None:
    fts = _intraday_two_session_panel()
    sig = fts.execution_lag_calendar_date(
        pd.Timestamp("2024-01-03 11:00:00"),
        lag=1,
        forbid_cross_session=True,
    )
    assert sig == pd.Timestamp("2024-01-03 10:00:00")


def test_execution_lag_cross_session_allowed_when_flag_off() -> None:
    fts = _intraday_two_session_panel()
    sig = fts.execution_lag_calendar_date(
        pd.Timestamp("2024-01-03 10:00:00"),
        lag=1,
        forbid_cross_session=False,
    )
    assert sig == pd.Timestamp("2024-01-02 11:00:00")


def test_trading_session_key_matches_midnight_anchor() -> None:
    fts = _intraday_two_session_panel()
    k1 = fts.trading_session_key(pd.Timestamp("2024-01-02 11:00:00"))
    k2 = fts.trading_session_key(pd.Timestamp("2024-01-02 10:00:00"))
    assert k1 == k2
    k3 = fts.trading_session_key(pd.Timestamp("2024-01-03 10:00:00"))
    assert k3 != k1


def test_finstrat_panel_date_for_execution_respects_flag() -> None:
    fts = _intraday_two_session_panel()

    def _algo(x):
        import jax.numpy as jnp

        return jnp.zeros(x.shape[0])

    fs_off = FinStrat(fts, _algo, signal_delay=1, intraday_session_isolated_lag=False)
    assert fs_off.panel_date_for_execution("2024-01-03 10:00:00") == pd.Timestamp(
        "2024-01-02 11:00:00"
    )

    fs_on = FinStrat(fts, _algo, signal_delay=1, intraday_session_isolated_lag=True)
    with pytest.raises(ValueError, match="cross_session_signal_lag"):
        fs_on.panel_date_for_execution("2024-01-03 10:00:00")
    assert fs_on.panel_date_for_execution("2024-01-03 11:00:00") == pd.Timestamp(
        "2024-01-03 10:00:00"
    )


def test_plot_concat_intraday_sessions_shape() -> None:
    import matplotlib

    matplotlib.use("Agg")
    fts = _intraday_two_session_panel()
    fig, ax = fts.plot_component_returns(
        show=False, concat_intraday_sessions=True, break_calendar_gaps=True
    )
    try:
        cal = fts.get_trading_calendar()
        series_lines = [ln for ln in ax.get_lines() if len(ln.get_xdata()) == len(cal)]
        assert len(series_lines) >= 1
        for ln in series_lines:
            assert np.array_equal(ln.get_xdata(), np.arange(len(cal)))
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_execution_lag_parity_minutes_and_hours() -> None:
    fts_h = _intraday_two_session_panel()
    fts_m = _intraday_two_session_panel_minutes()

    with pytest.raises(ValueError, match="cross_session_signal_lag"):
        fts_h.execution_lag_calendar_date(
            pd.Timestamp("2024-01-03 10:00:00"),
            lag=1,
            forbid_cross_session=True,
        )
    with pytest.raises(ValueError, match="cross_session_signal_lag"):
        fts_m.execution_lag_calendar_date(
            pd.Timestamp("2024-01-03 09:30:00"),
            lag=1,
            forbid_cross_session=True,
        )


def test_get_trading_calendar_legacy_instance_defaults_observed_mode() -> None:
    fts = _intraday_two_session_panel()
    delattr(fts, "_trading_axis_mode")
    cal = fts.get_trading_calendar()
    assert len(cal) == 4
