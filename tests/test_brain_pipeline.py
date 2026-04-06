"""BRAIN-style pipeline math: neutralization, scale, decay, golden vectors."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from shunya.algorithm.cross_section import neutralize_market, rank, winsorize
from shunya.algorithm.finstrat import FinStrat

from tests.conftest import make_stub_fints


def test_neutralize_market_ignores_nan_and_sum_zero():
    s = jnp.array([1.0, 2.0, float("nan"), 4.0], dtype=jnp.float32)
    c = neutralize_market(s)
    # finite mean = (1+2+4)/3
    assert float(jnp.sum(c)) == pytest.approx(0.0, abs=1e-5)
    assert float(c[2]) == 0.0


def test_gross_scaling_invariants_after_market_neutral():
    tickers = ["A", "B", "C", "D"]
    dates = ["2020-01-02", "2020-01-03"]
    fts = make_stub_fints(tickers, dates, base_price=50.0)
    cap = 20_000_000.0

    def algo(ctx) -> jnp.ndarray:
        # monotone scores 0 .. 1 like BRAIN rank output
        n = ctx.n_tickers
        return jnp.linspace(0.0, 1.0, n, dtype=jnp.float32)

    fs = FinStrat(
        fts,
        algo,
        neutralization="market",
        truncation=0.0,
    )
    names = fs.tickers_at("2020-01-03")
    notionals = fs.pass_(None, cap, tickers=names, execution_date="2020-01-03")
    arr = np.asarray(notionals, dtype=float)
    assert np.sum(arr) == pytest.approx(0.0, abs=1.0)
    assert np.sum(np.abs(arr)) == pytest.approx(cap, rel=1e-5)


def test_winsorize_all_nan_returns_zeros():
    x = jnp.array([float("nan"), float("nan")], dtype=jnp.float32)
    y = winsorize(x, 0.1)
    assert float(jnp.sum(jnp.abs(y))) == 0.0


def test_rank_sorts_non_finite_last():
    x = jnp.array([3.0, float("nan"), 1.0], dtype=jnp.float32)
    r = rank(x)
    assert float(r[1]) == 1.0
    assert float(r[2]) < float(r[1])


def test_linear_decay_matches_brain_weights():
    tickers = ["A", "B"]
    dates = ["2020-01-02", "2020-01-03", "2020-01-06"]
    fts = make_stub_fints(tickers, dates, base_price=100.0)

    def algo(ctx) -> jnp.ndarray:
        return jnp.ones((ctx.n_tickers,), dtype=jnp.float32)

    fs = FinStrat(
        fts,
        algo,
        decay_mode="linear",
        decay_window=3,
        decay=0.0,
        neutralization="none",
    )
    fs.reset_pipeline_state()
    n1 = fs.tickers_at(dates[0])
    o1 = fs.pass_(None, 1000.0, tickers=n1, execution_date=dates[0])
    n2 = fs.tickers_at(dates[1])
    o2 = fs.pass_(None, 1000.0, tickers=n2, execution_date=dates[1])
    n3 = fs.tickers_at(dates[2])
    o3 = fs.pass_(None, 1000.0, tickers=n3, execution_date=dates[2])
    # constant raw 1 → linear smoothed stays 1 when neutralization none and L1 scale
    assert np.allclose(np.asarray(o1), np.asarray(o2))
    assert np.allclose(np.asarray(o2), np.asarray(o3))


def test_signal_delay_shifts_panel_date():
    tickers = ["A", "B"]
    dates = ["2020-01-02", "2020-01-03"]
    fts = make_stub_fints(tickers, dates, base_price=10.0)
    fts.align_universe(("Close", "Volume"), on_bad_ticker="drop")

    def algo(ctx) -> jnp.ndarray:
        return ctx.close.latest.astype(jnp.float32)

    fs = FinStrat(
        fts,
        algo,
        signal_delay=1,
        neutralization="none",
    )
    assert fs.panel_date_for_execution("2020-01-03").normalize() == pd.Timestamp("2020-01-02").normalize()
    ctx = fs.context_at("2020-01-03", tickers=fs.tickers_at("2020-01-03"))
    assert float(ctx.close.latest[0]) == 10.0


def test_elapsed_trading_time_decay_weights_missing_business_bar_gap() -> None:
    fts = make_stub_fints(["A"], ["2020-01-02", "2020-01-06"], base_price=10.0)
    fts.df.loc[("A", pd.Timestamp("2020-01-02")), "Close"] = 1.0
    fts.df.loc[("A", pd.Timestamp("2020-01-06")), "Close"] = 0.0

    def algo(ctx) -> jnp.ndarray:
        return ctx.close.latest.astype(jnp.float32)

    fs_bar = FinStrat(
        fts,
        algo,
        decay_mode="ema",
        decay=0.5,
        temporal_mode="bar_step",
        neutralization="none",
    )
    fs_bar.pass_(
        None,
        1_000.0,
        tickers=["A"],
        execution_date="2020-01-02",
    )
    fs_bar.pass_(
        None,
        1_000.0,
        tickers=["A"],
        execution_date="2020-01-06",
    )

    fs_elapsed = FinStrat(
        fts,
        algo,
        decay_mode="ema",
        decay=0.5,
        temporal_mode="elapsed_trading_time",
        neutralization="none",
    )
    fs_elapsed.pass_(
        None,
        1_000.0,
        tickers=["A"],
        execution_date="2020-01-02",
    )
    fs_elapsed.pass_(
        None,
        1_000.0,
        tickers=["A"],
        execution_date="2020-01-06",
    )

    assert fs_bar._ema_prev["A"] == pytest.approx(0.5)
    assert fs_elapsed._ema_prev["A"] == pytest.approx(0.25)

