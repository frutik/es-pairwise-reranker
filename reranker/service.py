"""Reranking one query's hits: ask Jev about every pair of the first max_documents (cached answers are free), order
them by wins, and put the rest below. If Jev fails or the time budget runs out, every hit gets the same score instead,
so reranking can make search slower but never break it.

Equal scores mean "no opinion": a tie in wins, the fallback, and the hits past max_documents. They are exactly equal
on purpose. Elasticsearch sends the hits in index order, not in the search's order, and sends no scores, so this
service cannot know the search's order. Whoever calls it breaks equal scores with that order: in Elasticsearch, a
`linear` retriever that adds the original query with a tiny weight (elasticsearch/search_example.json). The results
list itself is sorted by score and then by input position, which is the search's order for callers that send the hits
in that order (rerank-eval)."""

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field

from reranker.blitz import blitz_sort
from reranker.cache import ComparisonCache
from reranker.config import Settings
from reranker.jev import JevClient, JevError
from reranker.questions import QuestionGenerator, questions_rubric
from reranker.ranking import RoundRobin, Winner, pairs, swapped

log = logging.getLogger(__name__)

# The fallback's score for every hit, and the score of every hit past max_documents (below the reranked, which are
# shares of wins in 0..1). Elasticsearch shifts scores before sorting (x + 1, or exp(x) when negative), keeping order.
_NO_OPINION = 0.0
_BELOW = -1.0


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
    input_tokens: int = 0  # what the asked comparisons cost, as Jev reported it
    cycles: int = 0
    reason: str | None = None
    beyond_window: int = 0  # hits past max_documents, not reranked
    ranker: str = "round-robin"
    rounds: int = 0  # Jev requests made one after another (round-robin: none, they all go at once)
    stats: dict = field(default_factory=dict)


def rubric(settings: Settings) -> str:
    """What makes two answers comparable: the same judge (URL and model) asked the same question. The URL is part of it
    so a fake Jev's answers never stand in for the real one's in a shared cache."""
    parts = [
        settings.jev_url,
        settings.jev_model,
        settings.instructions,
        str(settings.tie_band),
        str(settings.max_document_chars),
    ]
    return hashlib.sha256("\u0000".join(parts).encode()).hexdigest()


RANKERS = ("round-robin", "blitz", "decisions")


def blitz_rubric(settings: Settings) -> str:
    """BlitzRank asks each pair among the other documents of its group, not alone: another question, so its answers
    are kept apart from round-robin's."""
    return hashlib.sha256(f"blitz\u0000{rubric(settings)}".encode()).hexdigest()


class Reranker:
    def __init__(
        self, jev: JevClient, cache: ComparisonCache, settings: Settings, questions: QuestionGenerator | None = None
    ):
        self._jev = jev
        self._questions = questions
        self._questions_rubric = questions_rubric(settings)
        self._cache = cache
        self._settings = settings
        self._rubric = rubric(settings)
        self._blitz_rubric = blitz_rubric(settings)

    async def rerank(
        self, query: str, documents: list[str], top_n: int | None = None, ranker: str | None = None
    ) -> Outcome:
        ranker = ranker or self._settings.ranker
        if ranker not in RANKERS:
            raise ValueError(f"unknown ranker {ranker!r}; one of {', '.join(RANKERS)}")
        window = documents[: self._settings.max_documents]
        rank = {"round-robin": self._rank, "blitz": self._rank_blitz, "decisions": self._rank_decisions}[ranker]
        try:
            outcome = await asyncio.wait_for(rank(query, window), timeout=self._settings.time_budget_seconds)
        except TimeoutError:
            outcome = self._search_order(len(window), f"time budget of {self._settings.time_budget_seconds}s ran out")
        except JevError as error:
            outcome = self._search_order(len(window), str(error))
        if not outcome.reranked:
            log.warning("returning the search's order for %r: %s", query, outcome.reason)

        # Past the window: all equal, below everything reranked. Which hits fall here depends on the order they came
        # in, which from Elasticsearch is index order, so keep its rank_window_size at or below max_documents.
        rest = [Ranked(index, _BELOW) for index in range(len(window), len(documents))]
        if rest:
            outcome.beyond_window = len(rest)
            log.warning(
                "%d hits for %r are past max_documents (%d) and were not reranked",
                len(rest),
                query,
                self._settings.max_documents,
            )
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
        tokens = 0

        async def ask(i: int, j: int) -> None:
            nonlocal tokens
            async with limit:
                swap = swapped(query, window[i], window[j], self._rubric)
                comparison = await self._jev.compare(query, window[i], window[j], swapped=swap)
            answers[(i, j)] = comparison.winner
            tokens += comparison.input_tokens or 0
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
        results = [Ranked(index, ranked.scores[index] / others) for index in ranked.order]
        return Outcome(
            results=results,
            reranked=True,
            comparisons=len(answers),
            asked=len(missing),
            cycles=ranked.cycles,
            input_tokens=tokens,
        )

    async def _rank_blitz(self, query: str, window: list[str]) -> Outcome:
        count = len(window)
        every_pair = pairs(count)
        cached = await self._cache.get_many(
            query, [(window[i], window[j]) for i, j in every_pair], rubric=self._blitz_rubric
        )
        known = {pair: winner for pair, winner in zip(every_pair, cached, strict=True) if winner is not None}
        tokens = 0

        async def ask(group: list[int], missing: list[tuple[int, int]]) -> dict[tuple[int, int], Winner]:
            nonlocal tokens
            # Documents are labelled D1..Dn in an order fixed by their text, not by how good the graph thinks they
            # are, so that the judge's lean towards early labels does not favour the leaders.
            members = sorted(group, key=lambda position: self._shuffle_key(query, window[position]))
            local = {position: index for index, position in enumerate(members)}
            asked_pairs, flipped = [], []
            for i, j in missing:
                a, b = local[i], local[j]
                asked_pairs.append((min(a, b), max(a, b)))
                flipped.append(a > b)
            swaps = [swapped(query, window[members[a]], window[members[b]], self._blitz_rubric) for a, b in asked_pairs]
            winners, used = await self._jev.compare_group(
                query, [window[position] for position in members], asked_pairs, swaps
            )
            tokens += used
            answers = {}
            for (i, j), local_pair, flip in zip(missing, asked_pairs, flipped, strict=True):
                winner = winners[local_pair]
                if flip:
                    winner = {Winner.FIRST: Winner.SECOND, Winner.SECOND: Winner.FIRST}.get(winner, winner)
                answers[(i, j)] = winner
                await self._cache.put(query, window[i], window[j], winner, rubric=self._blitz_rubric)
            return answers

        settings = self._settings
        result = await blitz_sort(count, settings.blitz_k, settings.blitz_top_m, ask, known, settings.blitz_max_rounds)
        others = max(count - 1, 1)
        # The top: the share of the others not known to be better, equal for documents tied in a cycle. The rest of
        # the window: no opinion, below the top (BlitzRank does not rank past it).
        results = [Ranked(index, (others - better) / others) for index, better in result.top]
        results += [Ranked(index, _NO_OPINION - 0.5) for index in result.rest]
        return Outcome(
            results=results,
            reranked=True,
            comparisons=len(known) + result.asked,
            asked=result.asked,
            cycles=result.cycles,
            input_tokens=tokens,
            ranker="blitz",
            rounds=result.rounds,
            reason=None if result.certain else "stopped before the top was certain",
        )

    async def _rank_decisions(self, query: str, window: list[str]) -> Outcome:
        if len(window) < 2:
            results = [Ranked(index, _NO_OPINION) for index in range(len(window))]
            return Outcome(results=results, reranked=True, ranker="decisions")
        stats = {"llm_input_tokens": 0, "llm_output_tokens": 0, "questions_cached": True}
        questions = await self._cache.get_questions(self._questions_rubric, query)
        if questions is None:
            if self._questions is None:
                raise JevError("no question LLM configured")
            written = await self._questions.generate(query)
            questions = written.questions
            stats.update(
                llm_input_tokens=written.input_tokens, llm_output_tokens=written.output_tokens, questions_cached=False
            )
            await self._cache.put_questions(self._questions_rubric, query, questions)
        stats["questions"] = len(questions)

        limit = asyncio.Semaphore(self._settings.concurrency)

        async def decide(text: str) -> tuple[float, int]:
            async with limit:
                return await self._jev.decide(text, questions)

        answers = await asyncio.gather(*(decide(text) for text in window), return_exceptions=True)
        failures = [answer for answer in answers if isinstance(answer, BaseException)]
        if failures:
            raise failures[0] if isinstance(failures[0], JevError) else JevError(repr(failures[0]))
        # The article's score is the sum of P(yes); the mean orders the same and stays in 0..1, under the tie-break.
        scores = [total / len(questions) for total, _ in answers]
        order = sorted(range(len(window)), key=lambda index: (-scores[index], index))
        return Outcome(
            results=[Ranked(index, scores[index]) for index in order],
            reranked=True,
            asked=len(window),
            input_tokens=sum(tokens for _, tokens in answers),
            ranker="decisions",
            stats=stats,
        )

    def _shuffle_key(self, query: str, text: str) -> str:
        return hashlib.sha256("\u0000".join([self._blitz_rubric, query, text]).encode()).hexdigest()

    @staticmethod
    def _search_order(count: int, reason: str) -> Outcome:
        results = [Ranked(index, _NO_OPINION) for index in range(count)]
        return Outcome(results=results, reranked=False, reason=reason)
