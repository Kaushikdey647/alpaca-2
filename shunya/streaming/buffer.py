from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import Deque, Optional

import pandas as pd

from .events import MarketEvent


class SymbolRingBuffer:
    """Fixed-capacity FIFO event buffer for a single symbol."""

    def __init__(self, symbol: str, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.symbol = str(symbol)
        self.capacity = int(capacity)
        self._events: Deque[MarketEvent] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._events)

    def append(self, event: MarketEvent) -> None:
        if event.symbol != self.symbol:
            raise ValueError(
                f"buffer for {self.symbol!r} cannot ingest event for {event.symbol!r}"
            )
        self._events.append(event)

    def extend(self, events: Iterable[MarketEvent]) -> None:
        for event in events:
            self.append(event)

    def events(self) -> list[MarketEvent]:
        return list(self._events)

    def latest_event(self) -> Optional[MarketEvent]:
        if not self._events:
            return None
        return self._events[-1]

    def latest_price(self) -> Optional[float]:
        event = self.latest_event()
        if event is None:
            return None
        try:
            return float(event.ltp)
        except ValueError:
            return None

    def last_update_time(self) -> Optional[pd.Timestamp]:
        event = self.latest_event()
        return event.event_time if event is not None else None

    def is_stale(self, now: object, max_age: pd.Timedelta) -> bool:
        latest = self.last_update_time()
        if latest is None:
            return True
        return pd.Timestamp(now) - latest > max_age
