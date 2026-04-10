from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StreamingMetrics:
    """Minimal in-memory counters for the streaming pipeline."""

    events_seen: int = 0
    dropped_events: int = 0
    alpha_runs: int = 0
    orders_submitted: int = 0
    stale_symbols: int = 0
    queue_depths: dict[str, int] = field(default_factory=dict)
    last_alpha_latency_ms: float = 0.0
    last_order_latency_ms: float = 0.0

    def mark_queue_depth(self, symbol: str, depth: int) -> None:
        self.queue_depths[str(symbol)] = int(depth)
