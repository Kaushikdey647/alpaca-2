from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

import pandas as pd

EventKind = Literal["trade", "quote"]


def _coerce_timestamp(ts: object) -> pd.Timestamp:
    out = pd.Timestamp(ts)
    if out.tzinfo is not None:
        out = out.tz_convert("UTC").tz_localize(None)
    return out


@dataclass(frozen=True)
class MarketEvent:
    """Normalized market-data event used by the streaming pipeline."""

    symbol: str
    event_time: pd.Timestamp
    kind: EventKind
    price: Optional[float] = None
    size: float = 0.0
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None
    source: Optional[str] = None
    raw: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", str(self.symbol))
        object.__setattr__(self, "event_time", _coerce_timestamp(self.event_time))
        if self.kind not in ("trade", "quote"):
            raise ValueError(f"Unsupported market event kind: {self.kind!r}")
        if self.price is not None:
            object.__setattr__(self, "price", float(self.price))
        object.__setattr__(self, "size", float(self.size))
        if self.bid_price is not None:
            object.__setattr__(self, "bid_price", float(self.bid_price))
        if self.ask_price is not None:
            object.__setattr__(self, "ask_price", float(self.ask_price))
        if self.bid_size is not None:
            object.__setattr__(self, "bid_size", float(self.bid_size))
        if self.ask_size is not None:
            object.__setattr__(self, "ask_size", float(self.ask_size))

    @property
    def ltp(self) -> float:
        """Best-effort last-traded / synthetic price for alpha and OMS inputs."""
        if self.price is not None:
            return float(self.price)
        if self.bid_price is not None and self.ask_price is not None:
            return 0.5 * (float(self.bid_price) + float(self.ask_price))
        if self.bid_price is not None:
            return float(self.bid_price)
        if self.ask_price is not None:
            return float(self.ask_price)
        raise ValueError(f"Event for {self.symbol!r} does not carry a usable price")


def trade_event(
    symbol: str,
    event_time: object,
    price: float,
    *,
    size: float = 0.0,
    source: Optional[str] = None,
    raw: Any = None,
) -> MarketEvent:
    return MarketEvent(
        symbol=symbol,
        event_time=_coerce_timestamp(event_time),
        kind="trade",
        price=price,
        size=size,
        source=source,
        raw=raw,
    )


def quote_event(
    symbol: str,
    event_time: object,
    *,
    bid_price: Optional[float] = None,
    ask_price: Optional[float] = None,
    bid_size: Optional[float] = None,
    ask_size: Optional[float] = None,
    source: Optional[str] = None,
    raw: Any = None,
) -> MarketEvent:
    return MarketEvent(
        symbol=symbol,
        event_time=_coerce_timestamp(event_time),
        kind="quote",
        bid_price=bid_price,
        ask_price=ask_price,
        bid_size=bid_size,
        ask_size=ask_size,
        source=source,
        raw=raw,
    )
