from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class UniverseCandidate:
    symbol: str
    score: float = 0.0


class UniverseSelector:
    """Select the highest-priority symbols under a subscription budget."""

    @staticmethod
    def select(
        candidates: Sequence[str] | Sequence[UniverseCandidate],
        *,
        limit: int,
        scores: Mapping[str, float] | None = None,
    ) -> list[str]:
        if limit < 1:
            return []

        rows: list[UniverseCandidate] = []
        for idx, candidate in enumerate(candidates):
            if isinstance(candidate, UniverseCandidate):
                rows.append(candidate)
                continue
            symbol = str(candidate)
            score = 0.0 if scores is None else float(scores.get(symbol, 0.0))
            rows.append(UniverseCandidate(symbol=symbol, score=score - idx * 1e-12))

        ranked = sorted(
            rows,
            key=lambda row: (-float(row.score), row.symbol),
        )
        return [row.symbol for row in ranked[: int(limit)]]
