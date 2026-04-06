"""Tests for :class:`FinStrat`."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from shunya.algorithm.finstrat import FinStrat

from tests.conftest import make_stub_fints


def test_pass_market_neutral_sums_to_zero_and_hits_gross():
    tickers = ["AAA", "BBB", "CCC"]
    dates = ["2020-01-02", "2020-01-03"]
    fts = make_stub_fints(tickers, dates, base_price=50.0)
    for i, (t, d) in enumerate(fts.df.index):
        fts.df.loc[(t, d), "Close"] = 50.0 + float(i)

    def algo(ctx) -> jnp.ndarray:
        return ctx.close.latest.astype(jnp.float32)

    fs = FinStrat(
        fts,
        algo,
        neutralization="market",
        decay=0.0,
        truncation=0.0,
    )
    names = fs.tickers_at("2020-01-03")
    notionals = fs.pass_(None, 30_000.0, tickers=names, execution_date="2020-01-03")
    assert float(np.sum(np.asarray(notionals))) == pytest.approx(0.0, abs=5e-3)
    assert float(np.sum(np.abs(np.asarray(notionals)))) == pytest.approx(30_000.0, rel=1e-4)


def test_group_labels_at_reads_sector_and_industry_columns():
    tickers = ["AAA", "BBB", "CCC"]
    dates = ["2020-01-02", "2020-01-03"]
    fts = make_stub_fints(tickers, dates, base_price=50.0)
    d = "2020-01-03"
    mapping = {
        "AAA": ("Tech", "Semis"),
        "BBB": ("Tech", "Software"),
        "CCC": ("Energy", "OilGas"),
    }
    for t in tickers:
        fts.df.loc[(t, d), "Sector"] = mapping[t][0]
        fts.df.loc[(t, d), "Industry"] = mapping[t][1]

    fs = FinStrat(
        fts,
        lambda ctx: ctx.close.latest.astype(jnp.float32),
        neutralization="group",
    )
    names = fs.tickers_at("2020-01-03")
    sec = fs.group_labels_at("2020-01-03", names, "Sector")
    ind = fs.group_labels_at("2020-01-03", names, "Industry")
    assert sec.shape[0] == len(names)
    assert ind.shape[0] == len(names)
