from __future__ import annotations

from typing import Dict, Iterable, Optional

from .buffer import SymbolRingBuffer
from .events import MarketEvent


class StreamingState:
    """Registry of active symbols and their rolling event buffers."""

    def __init__(self, lookback: int) -> None:
        if lookback < 1:
            raise ValueError("lookback must be >= 1")
        self.lookback = int(lookback)
        self._buffers: Dict[str, SymbolRingBuffer] = {}
        self._active_symbols: list[str] = []

    def ensure_symbol(self, symbol: str) -> SymbolRingBuffer:
        name = str(symbol)
        buf = self._buffers.get(name)
        if buf is None:
            buf = SymbolRingBuffer(name, self.lookback)
            self._buffers[name] = buf
        if name not in self._active_symbols:
            self._active_symbols.append(name)
        return buf

    def set_active_symbols(self, symbols: Iterable[str]) -> list[str]:
        ordered = []
        seen = set()
        for symbol in symbols:
            name = str(symbol)
            if name in seen:
                continue
            self.ensure_symbol(name)
            ordered.append(name)
            seen.add(name)
        self._active_symbols = ordered
        return list(self._active_symbols)

    def active_symbols(self) -> list[str]:
        return list(self._active_symbols)

    def ingest(self, event: MarketEvent) -> None:
        self.ensure_symbol(event.symbol).append(event)

    def buffer(self, symbol: str) -> Optional[SymbolRingBuffer]:
        return self._buffers.get(str(symbol))

    def latest_prices(self, symbols: Optional[Iterable[str]] = None) -> dict[str, float]:
        names = self._active_symbols if symbols is None else [str(s) for s in symbols]
        out: dict[str, float] = {}
        for symbol in names:
            buf = self._buffers.get(symbol)
            if buf is None:
                continue
            price = buf.latest_price()
            if price is not None:
                out[symbol] = float(price)
        return out
