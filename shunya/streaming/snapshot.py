from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, Literal, Optional

import numpy as np
import pandas as pd

from .events import MarketEvent

FillPolicy = Literal["ffill", "nan"]


def _coerce_interval(value: object) -> pd.Timedelta:
    out = pd.Timedelta(value)
    if out <= pd.Timedelta(0):
        raise ValueError("bar_interval must be positive")
    return out


def _bucket_start(ts: pd.Timestamp, interval: pd.Timedelta) -> pd.Timestamp:
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    ns = interval.value
    return pd.Timestamp((ts.value // ns) * ns)


@dataclass
class MicroBar:
    symbol: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    event_count: int = 1
    complete: bool = False

    def update(self, event: MarketEvent) -> None:
        price = float(event.ltp)
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.volume += float(event.size)
        self.event_count += 1


@dataclass(frozen=True)
class StreamingSnapshot:
    symbols: list[str]
    timestamps: pd.DatetimeIndex
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    adj_volume: np.ndarray
    features: dict[str, np.ndarray] = field(default_factory=dict)


class MicroBarAggregator:
    """Aggregate asynchronous events into per-symbol micro-bars."""

    def __init__(self, *, bar_interval: object, lookback: int) -> None:
        if lookback < 1:
            raise ValueError("lookback must be >= 1")
        self.bar_interval = _coerce_interval(bar_interval)
        self.lookback = int(lookback)
        self._current: Dict[str, MicroBar] = {}
        self._history: Dict[str, Deque[MicroBar]] = defaultdict(
            lambda: deque(maxlen=self.lookback)
        )

    def observe(self, event: MarketEvent) -> Optional[MicroBar]:
        symbol = event.symbol
        start = _bucket_start(event.event_time, self.bar_interval)
        end = start + self.bar_interval
        current = self._current.get(symbol)
        if current is None or current.start_time != start:
            completed: Optional[MicroBar] = None
            if current is not None:
                current.complete = True
                self._history[symbol].append(current)
                completed = current
            price = float(event.ltp)
            self._current[symbol] = MicroBar(
                symbol=symbol,
                start_time=start,
                end_time=end,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=float(event.size),
            )
            return completed
        current.update(event)
        return None

    def bars_for(self, symbol: str, *, include_incomplete: bool = True) -> list[MicroBar]:
        out = list(self._history.get(str(symbol), ()))
        current = self._current.get(str(symbol))
        if include_incomplete and current is not None:
            out.append(current)
        return out

    def latest_prices(self, symbols: Optional[Iterable[str]] = None) -> dict[str, float]:
        names = list(self._current) if symbols is None else [str(s) for s in symbols]
        out: dict[str, float] = {}
        for symbol in names:
            current = self._current.get(symbol)
            if current is not None:
                out[symbol] = float(current.close)
        return out


class SnapshotBuilder:
    """Materialize a rectangular `(time, n_tickers)` snapshot from micro-bars."""

    def __init__(self, *, lookback: int, fill_policy: FillPolicy = "ffill") -> None:
        if lookback < 1:
            raise ValueError("lookback must be >= 1")
        if fill_policy not in ("ffill", "nan"):
            raise ValueError("fill_policy must be 'ffill' or 'nan'")
        self.lookback = int(lookback)
        self.fill_policy = fill_policy

    def build(
        self,
        aggregator: MicroBarAggregator,
        symbols: Iterable[str],
        *,
        include_incomplete: bool = True,
    ) -> StreamingSnapshot:
        names = [str(symbol) for symbol in symbols]
        if not names:
            raise ValueError("symbols must be non-empty")

        by_symbol: dict[str, list[MicroBar]] = {
            name: aggregator.bars_for(name, include_incomplete=include_incomplete)
            for name in names
        }
        timestamps = sorted(
            {
                bar.end_time
                for bars in by_symbol.values()
                for bar in bars
            }
        )
        if not timestamps:
            raise ValueError("No micro-bars available for snapshot")
        timestamps = timestamps[-self.lookback :]
        index = pd.DatetimeIndex(timestamps)

        open_mat = np.full((len(index), len(names)), np.nan, dtype=float)
        high_mat = np.full((len(index), len(names)), np.nan, dtype=float)
        low_mat = np.full((len(index), len(names)), np.nan, dtype=float)
        close_mat = np.full((len(index), len(names)), np.nan, dtype=float)
        volume_mat = np.zeros((len(index), len(names)), dtype=float)

        for col, symbol in enumerate(names):
            bars = by_symbol[symbol]
            row_map = {bar.end_time: bar for bar in bars if bar.end_time in index}
            last_close = np.nan
            for row, ts in enumerate(index):
                bar = row_map.get(ts)
                if bar is None:
                    if self.fill_policy == "ffill" and np.isfinite(last_close):
                        open_mat[row, col] = last_close
                        high_mat[row, col] = last_close
                        low_mat[row, col] = last_close
                        close_mat[row, col] = last_close
                    continue
                open_mat[row, col] = float(bar.open)
                high_mat[row, col] = float(bar.high)
                low_mat[row, col] = float(bar.low)
                close_mat[row, col] = float(bar.close)
                volume_mat[row, col] = float(bar.volume)
                last_close = float(bar.close)

        return StreamingSnapshot(
            symbols=names,
            timestamps=index,
            open=open_mat,
            high=high_mat,
            low=low_mat,
            close=close_mat,
            adj_volume=volume_mat,
        )
