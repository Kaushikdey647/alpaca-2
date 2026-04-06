"""
Example alpha collection using the context-based API.

Each alpha follows ``def alpha(ctx) -> jnp.ndarray`` and can be passed directly
to ``FinStrat``.
"""

from .breakout_20 import alpha as breakout_20
from .mean_reversion_5 import alpha as mean_reversion_5
from .sma_ratio_50 import alpha as sma_ratio_50
from .volume_price_trend_20 import alpha as volume_price_trend_20

ALL_ALPHAS = {
    "sma_ratio_50": sma_ratio_50,
    "mean_reversion_5": mean_reversion_5,
    "breakout_20": breakout_20,
    "volume_price_trend_20": volume_price_trend_20,
}

__all__ = [
    "ALL_ALPHAS",
    "sma_ratio_50",
    "mean_reversion_5",
    "breakout_20",
    "volume_price_trend_20",
]

