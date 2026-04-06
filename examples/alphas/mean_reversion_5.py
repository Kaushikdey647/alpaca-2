from __future__ import annotations

import jax.numpy as jnp


def alpha(ctx) -> jnp.ndarray:
    """
    Short-horizon mean reversion: rank(-(close / SMA5 - 1)).
    """
    sma_5 = ctx.ts.mean(ctx.close, 5)
    deviation = (ctx.close / sma_5) - 1.0
    return ctx.cs.rank(-deviation)

