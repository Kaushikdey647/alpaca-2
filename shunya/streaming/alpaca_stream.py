from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

import pandas as pd

from .events import MarketEvent, quote_event, trade_event
from .subscriptions import SubscriptionBackend


def _attr(obj: Any, *names: str) -> Any:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def normalize_alpaca_trade(message: Any) -> MarketEvent:
    ts = _attr(message, "timestamp", "t")
    symbol = _attr(message, "symbol", "S")
    price = _attr(message, "price", "p")
    size = _attr(message, "size", "s") or 0.0
    if ts is None or symbol is None or price is None:
        raise ValueError("trade message missing timestamp, symbol, or price")
    return trade_event(
        str(symbol),
        pd.Timestamp(ts),
        float(price),
        size=float(size),
        source="alpaca_trade",
        raw=message,
    )


def normalize_alpaca_quote(message: Any) -> MarketEvent:
    ts = _attr(message, "timestamp", "t")
    symbol = _attr(message, "symbol", "S")
    bid_price = _attr(message, "bid_price", "bp")
    ask_price = _attr(message, "ask_price", "ap")
    bid_size = _attr(message, "bid_size", "bs")
    ask_size = _attr(message, "ask_size", "as_")
    if ask_size is None:
        ask_size = _attr(message, "ask_size", "as")
    if ts is None or symbol is None:
        raise ValueError("quote message missing timestamp or symbol")
    return quote_event(
        str(symbol),
        pd.Timestamp(ts),
        bid_price=None if bid_price is None else float(bid_price),
        ask_price=None if ask_price is None else float(ask_price),
        bid_size=None if bid_size is None else float(bid_size),
        ask_size=None if ask_size is None else float(ask_size),
        source="alpaca_quote",
        raw=message,
    )


class AlpacaStreamClient(SubscriptionBackend):
    """
    Thin Alpaca websocket wrapper with broker-neutral event normalization.

    The dependency is imported lazily so the rest of the package can be imported in
    environments without `alpaca-py`.
    """

    def __init__(
        self,
        *,
        api_key: str,
        secret_key: str,
        feed: Optional[str] = None,
        raw_data: bool = False,
    ) -> None:
        try:
            from alpaca.data.live.stock import StockDataStream
        except ImportError as exc:
            raise ImportError(
                "alpaca-py with stock live data support is required for AlpacaStreamClient"
            ) from exc

        self._stream = StockDataStream(
            api_key,
            secret_key,
            raw_data=raw_data,
            feed=feed,
        )
        self._trade_symbols: set[str] = set()
        self._quote_symbols: set[str] = set()
        self._on_trade: Optional[Callable[[MarketEvent], None]] = None
        self._on_quote: Optional[Callable[[MarketEvent], None]] = None

    def set_handlers(
        self,
        *,
        on_trade: Optional[Callable[[MarketEvent], None]] = None,
        on_quote: Optional[Callable[[MarketEvent], None]] = None,
    ) -> None:
        self._on_trade = on_trade
        self._on_quote = on_quote

    def subscribe(
        self,
        symbols: list[str],
        *,
        on_trade: Optional[Callable[[MarketEvent], None]] = None,
        on_quote: Optional[Callable[[MarketEvent], None]] = None,
    ) -> None:
        names = [str(symbol) for symbol in symbols]
        if not names:
            return
        trade_cb = on_trade or self._on_trade
        quote_cb = on_quote or self._on_quote

        if trade_cb is not None and hasattr(self._stream, "subscribe_trades"):
            async def _trade_handler(message: Any) -> None:
                trade_cb(normalize_alpaca_trade(message))

            self._stream.subscribe_trades(_trade_handler, *names)
            self._trade_symbols.update(names)

        if quote_cb is not None and hasattr(self._stream, "subscribe_quotes"):
            async def _quote_handler(message: Any) -> None:
                quote_cb(normalize_alpaca_quote(message))

            self._stream.subscribe_quotes(_quote_handler, *names)
            self._quote_symbols.update(names)

    def unsubscribe(self, symbols: list[str]) -> None:
        names = [str(symbol) for symbol in symbols]
        if not names:
            return
        if hasattr(self._stream, "unsubscribe_trades"):
            self._stream.unsubscribe_trades(*names)
        if hasattr(self._stream, "unsubscribe_quotes"):
            self._stream.unsubscribe_quotes(*names)
        self._trade_symbols.difference_update(names)
        self._quote_symbols.difference_update(names)

    def current_subscriptions(self) -> list[str]:
        return sorted(self._trade_symbols | self._quote_symbols)

    def run(self) -> None:
        self._stream.run()

    def stop(self) -> None:
        stopper = getattr(self._stream, "stop", None)
        if stopper is not None:
            stopper()
