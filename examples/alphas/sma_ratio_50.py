from __future__ import annotations

import jax.numpy as jnp


def alpha(ctx) -> jnp.ndarray:
    """
    Trend alpha: rank(close / SMA50).
    """
    sma_50 = ctx.ts.mean(ctx.close, 50)
    signal = ctx.close / sma_50
    return ctx.cs.rank(signal)

