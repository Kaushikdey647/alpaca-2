from __future__ import annotations

import jax.numpy as jnp


def alpha(ctx) -> jnp.ndarray:
    """
    Volume-confirmed trend alpha.
    """
    price_trend = ctx.close / ctx.ts.mean(ctx.close, 20)
    vol_trend = ctx.adj_volume / ctx.ts.mean(ctx.adj_volume, 20)
    signal = price_trend * vol_trend
    return ctx.cs.rank(signal)

