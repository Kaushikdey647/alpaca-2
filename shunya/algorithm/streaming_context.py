from __future__ import annotations

import jax.numpy as jnp

from .alpha_context import AlphaContext
from ..streaming.snapshot import StreamingSnapshot


class StreamingContextBuilder:
    """Convert streaming snapshots into the regular alpha context contract."""

    @staticmethod
    def build(snapshot: StreamingSnapshot) -> AlphaContext:
        return AlphaContext(
            open=jnp.asarray(snapshot.open, dtype=jnp.float32),
            high=jnp.asarray(snapshot.high, dtype=jnp.float32),
            low=jnp.asarray(snapshot.low, dtype=jnp.float32),
            close=jnp.asarray(snapshot.close, dtype=jnp.float32),
            adj_volume=jnp.asarray(snapshot.adj_volume, dtype=jnp.float32),
            features={
                name: jnp.asarray(value, dtype=jnp.float32)
                for name, value in snapshot.features.items()
            },
        )
