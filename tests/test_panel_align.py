"""Tests for :meth:`finTs.align_universe` and trading calendar helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shunya.data.fints import PanelAlignReport, finTs
from shunya.data.timeframes import BarSpec, BarUnit, default_bar_index_policy, default_bar_spec

from tests.conftest import make_stub_fints


def test_align_universe_keeps_complete_tickers_and_sets_calendar():
    tickers = ["AAA", "BBB", "CCC"]
    dates = ["2020-01-02", "2020-01-03", "2020-01-06"]
    fts = make_stub_fints(tickers, dates, base_price=100.0)
    fts.df.loc[("CCC", "2020-01-03"), "Close"] = np.nan

    rep = fts.align_universe(
        ("Open", "High", "Low", "Close", "Volume"),
        on_bad_ticker="drop",
    )
    assert isinstance(rep, PanelAlignReport)
    assert "CCC" in rep.dropped_tickers
    assert fts.ticker_list == ["AAA", "BBB"]
    assert fts._aligned_calendar is not None
    assert len(fts._aligned_calendar) == len(dates)


def test_align_universe_raise_on_bad_ticker():
    tickers = ["AAA", "BBB"]
    dates = ["2020-01-02", "2020-01-03"]
    fts = make_stub_fints(tickers, dates, base_price=50.0)
    fts.df.loc[("BBB", "2020-01-03"), "Volume"] = np.nan

    with pytest.raises(ValueError, match="on_bad_ticker='raise'"):
        fts.align_universe(
            ("Close", "Volume"),
            on_bad_ticker="raise",
        )


def test_execution_lag_calendar_date():
    tickers = ["AAA", "BBB"]
    dates = ["2020-01-02", "2020-01-03", "2020-01-06"]
    fts = make_stub_fints(tickers, dates, base_price=10.0)
    fts.align_universe(("Close", "Volume"), on_bad_ticker="drop")

    d1 = fts.execution_lag_calendar_date("2020-01-06", lag=1)
    assert d1.normalize() == pd.Timestamp("2020-01-03").normalize()


def test_align_universe_intraday_intersects_on_exact_timestamps():
    t0 = pd.Timestamp("2020-01-02 09:30:00")
    t1 = pd.Timestamp("2020-01-02 09:35:00")
    rows = [
        ("AAA", t0),
        ("AAA", t1),
        ("BBB", t0),
        ("BBB", t1),
    ]
    idx = pd.MultiIndex.from_tuples(rows, names=["Ticker", "Date"])
    df = pd.DataFrame(
        {
            "Open": [100.0] * len(idx),
            "High": [101.0] * len(idx),
            "Low": [99.0] * len(idx),
            "Close": [100.0] * len(idx),
            "Volume": [1e6] * len(idx),
        },
        index=idx,
    )
    stub = object.__new__(finTs)
    stub.start_date = "2020-01-02"
    stub.end_date = "2020-01-02"
    stub.session = None
    stub.ticker_list = ["AAA", "BBB"]
    stub.df = df
    stub._aligned_calendar = None
    stub.bar_spec = BarSpec(BarUnit.MINUTES, 5)
    stub._bar_index_policy = default_bar_index_policy()
    stub._trading_axis_mode = "observed"

    rep = stub.align_universe(("Open", "High", "Low", "Close", "Volume"))
    assert len(rep.calendar) == 2
    assert rep.calendar[0] == t0
    assert rep.calendar[1] == t1


def test_finbt_raises_when_aligned_calendar_has_hole_for_one_ticker():
    """``_aligned_calendar`` can be wider than a ticker's rows; reindex must fail."""
    import jax.numpy as jnp

    from shunya.algorithm.finbt import FinBT
    from shunya.algorithm.finstrat import FinStrat

    d1 = pd.Timestamp("2020-01-02").normalize()
    d2 = pd.Timestamp("2020-01-03").normalize()
    d3 = pd.Timestamp("2020-01-06").normalize()
    rows = [
        ("AAA", d1),
        ("AAA", d2),
        ("AAA", d3),
        ("BBB", d1),
        ("BBB", d2),
    ]
    idx = pd.MultiIndex.from_tuples(rows, names=["Ticker", "Date"])
    df = pd.DataFrame(
        {
            "Open": [100.0] * len(idx),
            "High": [101.0] * len(idx),
            "Low": [99.0] * len(idx),
            "Close": [100.0] * len(idx),
            "Volume": [1e6] * len(idx),
        },
        index=idx,
    )
    stub = object.__new__(finTs)
    stub.start_date = str(d1.date())
    stub.end_date = str(d3.date())
    stub.session = None
    stub.ticker_list = ["AAA", "BBB"]
    stub.df = df
    stub._aligned_calendar = pd.DatetimeIndex([d1, d2, d3])
    stub.bar_spec = default_bar_spec()
    stub._bar_index_policy = default_bar_index_policy()
    stub._trading_axis_mode = "observed"

    def algo(ctx) -> jnp.ndarray:
        return ctx.close.latest.astype(jnp.float32)

    fs = FinStrat(
        stub,
        algo,
        neutralization="market",
    )
    with pytest.raises(ValueError, match="missing OHLCV"):
        FinBT(fs, stub, cash=10_000.0).run()
