from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable


@runtime_checkable
class SubscriptionBackend(Protocol):
    def subscribe(self, symbols: list[str]) -> None: ...

    def unsubscribe(self, symbols: list[str]) -> None: ...

    def current_subscriptions(self) -> list[str]: ...


class SubscriptionManager:
    """Reconcile desired symbol subscriptions against the broker stream backend."""

    def __init__(self, backend: SubscriptionBackend, *, max_symbols: int) -> None:
        if max_symbols < 1:
            raise ValueError("max_symbols must be >= 1")
        self._backend = backend
        self._max_symbols = int(max_symbols)

    @property
    def max_symbols(self) -> int:
        return self._max_symbols

    def rebalance(self, desired_symbols: Iterable[str]) -> list[str]:
        desired = []
        seen = set()
        for symbol in desired_symbols:
            name = str(symbol)
            if name in seen:
                continue
            desired.append(name)
            seen.add(name)
            if len(desired) >= self._max_symbols:
                break

        current = self._backend.current_subscriptions()
        current_set = set(current)
        desired_set = set(desired)

        to_unsubscribe = sorted(current_set - desired_set)
        if to_unsubscribe:
            self._backend.unsubscribe(to_unsubscribe)

        to_subscribe = [symbol for symbol in desired if symbol not in current_set]
        if to_subscribe:
            self._backend.subscribe(to_subscribe)

        return desired
