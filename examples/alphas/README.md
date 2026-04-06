# Example Alphas

This folder contains context-style alpha examples for `FinStrat`.

## Contract

Each alpha function follows:

```python
def alpha(ctx) -> jnp.ndarray:
    ...
```

Where `ctx` provides:

- `ctx.open`, `ctx.high`, `ctx.low`, `ctx.close`, `ctx.adj_volume`
- `ctx.ts.*` time-series operators
- `ctx.cs.*` cross-sectional operators

## Included

- `sma_ratio_50`: trend via `close / SMA(50)`
- `mean_reversion_5`: short-term reversion via negative 5-bar deviation
- `breakout_20`: 20-bar momentum via delayed-close ratio
- `volume_price_trend_20`: trend weighted by relative volume

## Usage

```python
from shunya import FinStrat, finTs
from examples.alphas import sma_ratio_50

fts = finTs("2023-01-01", "2024-01-01", ["AAPL", "MSFT", "NVDA"])
fs = FinStrat(fts, sma_ratio_50, neutralization="market")
```

