"""Reranking one query's hits: ask Jev about every pair of the first max_documents (cached answers are free), order
them by wins, and keep the rest below in the search's order. If Jev fails or the time budget runs out, the search's
own order comes back instead, so reranking can make search slower but never break it."""

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field

from reranker.cache import ComparisonCache
from reranker.config import Settings
from reranker.jev import JevClient, JevError
from reranker.ranking import RoundRobin, Winner, pairs, swapped

log = logging.getLogger(__name__)

# Keeps equal win counts in the search's order once Elasticsearch sorts by score.
_TIEBREAK = 1e-6


@dataclass
class Ranked:
    index: int
    score: float


@dataclass
class Outcome:
    results: list[Ranked]
    reranked: bool  # False when the search's order came back instead (fail open)
    comparisons: int = 0
    asked: int = 0  # comparisons that were not in the cache
    cycles: int = 0
    reason: str | None = None
    stats: dict = field(default_factory=dict)


def rubric(settings: Settings) -> str:
    """What makes two answers comparable: the same model asked the same question."""
    parts = [settings.jev_model, settings.instructions, str(settings.tie_band), str(settings.max_document_chars)]
    return hashlib.sha256("\u0000".join(parts).encode()).hexdigest()


class Reranker:
    def __init__(self, jev: JevClient, cache: ComparisonCache, settings: Settings):
        self._jev = jev
        self._cache = cache
        self._settings = settings
        self._rubric = rubric(settings)

    async def rerank(self, query: str, documents: list[str], top_n: int | None = None) -> Outcome:
        window = documents[: self._settings.max_documents]
        try:
            outcome = await asyncio.wait_for(self._rank(query, window), timeout=self._settings.time_budget_seconds)
        except TimeoutError:
            outcome = self._search_order(len(window), f"time budget of {self._settings.time_budget_seconds}s ran out")
        except JevError as error:
            outcome = self._search_order(len(window), str(error))
        if not outcome.reranked:
            log.warning("returning the search's order for %r: %s", query, outcome.reason)

        # Past the window, the search's order, below everything reranked.
        rest = [
            Ranked(index, -1.0 - position * _TIEBREAK)
            for position, index in enumerate(range(len(window), len(documents)))
        ]
        outcome.results = (outcome.results + rest)[: top_n or None]
        return outcome

    async def _rank(self, query: str, window: list[str]) -> Outcome:
        answers: dict[tuple[int, int], Winner] = {}
        missing: list[tuple[int, int]] = []
        every_pair = pairs(len(window))
        cached = await self._cache.get_many(query, [(window[i], window[j]) for i, j in every_pair])
        for (i, j), winner in zip(every_pair, cached, strict=True):
            if winner is None:
                missing.append((i, j))
            else:
                answers[(i, j)] = winner

        limit = asyncio.Semaphore(self._settings.concurrency)

        async def ask(i: int, j: int) -> None:
            async with limit:
                swap = swapped(query, window[i], window[j], self._rubric)
                comparison = await self._jev.compare(query, window[i], window[j], swapped=swap)
            answers[(i, j)] = comparison.winner
            await self._cache.put(query, window[i], window[j], comparison.winner)

        failures = [
            result
            for result in await asyncio.gather(*(ask(i, j) for i, j in missing), return_exceptions=True)
            if isinstance(result, BaseException)
        ]
        if failures:
            raise failures[0] if isinstance(failures[0], JevError) else JevError(repr(failures[0]))

        ranked = RoundRobin.from_answers(len(window), answers)
        others = max(len(window) - 1, 1)
        results = [
            Ranked(index, ranked.scores[index] / others - position * _TIEBREAK)
            for position, index in enumerate(ranked.order)
        ]
        return Outcome(
            results=results, reranked=True, comparisons=len(answers), asked=len(missing), cycles=ranked.cycles
        )

    @staticmethod
    def _search_order(count: int, reason: str) -> Outcome:
        results = [Ranked(index, 1.0 - index * _TIEBREAK) for index in range(count)]
        return Outcome(results=results, reranked=False, reason=reason)
