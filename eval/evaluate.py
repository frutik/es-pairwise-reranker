"""Does reranking agree with people better than the search's own order? Reranks judged queries with the real judge and
reports nDCG@10, Kendall's tau-b and pairwise accuracy for both orders, as the Quepid experiments did.

    rerank-eval judged.jsonl [--limit 30] [--per-query]

Each line of the input is one query, its hits in the search's order, each with a human grade:

    {"query": "angle bracket", "documents": [{"text": "Simpson Strong-Tie ...", "grade": 3}, ...]}

It calls Jev (RERANKER_JEV_API_KEY and the other settings apply), so it costs what the comparisons cost.

Besides the means it reports, per measure, on how many queries reranking was better, equal or worse, and a 95% paired
bootstrap interval for the mean difference (reranked minus search). An interval above zero is a win that is not
noise; one that spans zero is not a result yet.
"""

import argparse
import asyncio
import json
import math
import random
import statistics
import sys
import time

import httpx

from reranker.cache import open_cache
from reranker.config import Settings
from reranker.jev import JevClient
from reranker.metrics import kendall_tau_b, ndcg, pairwise_accuracy
from reranker.questions import QuestionGenerator
from reranker.service import Reranker, rubric

MEASURES = {"nDCG@10": ndcg, "tau-b": kendall_tau_b, "pairwise accuracy": pairwise_accuracy}


def measures_at(depth: int) -> dict:
    """The measures on the first `depth` hits of an order of more candidates: nDCG@depth against the ideal order of all
    of them (so bringing a better hit up from below counts), tau-b and pairwise accuracy among the hits shown."""
    return {
        f"nDCG@{depth}": lambda grades: ndcg(grades, depth),
        f"tau-b@{depth}": lambda grades: kendall_tau_b(grades[:depth]),
        f"pairwise acc.@{depth}": lambda grades: pairwise_accuracy(grades[:depth]),
    }


async def evaluate(
    queries: list[dict],
    settings: Settings,
    out=sys.stdout,
    per_query: bool = False,
    ranker: str | None = None,
    depth: int | None = None,
) -> dict:
    async with (
        httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client,
        open_cache(settings, rubric(settings)) as cache,
    ):
        jev = JevClient(
            client,
            url=settings.jev_url,
            api_key=settings.jev_api_key,
            model=settings.jev_model,
            instructions=settings.instructions,
            tie_band=settings.tie_band,
            max_document_chars=settings.max_document_chars,
        )
        reranker = Reranker(jev, cache, settings, QuestionGenerator(client, settings))
        return await evaluate_with(reranker, queries, out, per_query, ranker, depth)


async def evaluate_with(
    reranker: Reranker,
    queries: list[dict],
    out=sys.stdout,
    per_query: bool = False,
    ranker: str | None = None,
    depth: int | None = None,
) -> dict:
    measures = MEASURES if depth is None else measures_at(depth)
    tau = next(name for name in measures if name.startswith("tau-b"))
    scores = {"search": {name: [] for name in measures}, "reranked": {name: [] for name in measures}}
    fell_back = comparisons = tokens = llm_input = llm_output = 0
    seconds: list[float] = []
    for line in queries:
        texts = [document["text"] for document in line["documents"]]
        grades = [float(document["grade"]) for document in line["documents"]]
        started = time.perf_counter()
        outcome = await reranker.rerank(line["query"], texts, ranker=ranker)
        seconds.append(time.perf_counter() - started)
        fell_back += not outcome.reranked
        comparisons += outcome.asked
        tokens += outcome.input_tokens
        llm_input += outcome.stats.get("llm_input_tokens", 0)
        llm_output += outcome.stats.get("llm_output_tokens", 0)
        reranked_grades = [grades[ranked.index] for ranked in outcome.results]
        for name, measure in measures.items():
            scores["search"][name].append(measure(grades))
            scores["reranked"][name].append(measure(reranked_grades))
        if per_query:
            taus = (scores["search"][tau][-1], scores["reranked"][tau][-1])
            print(
                f"{line['query'][:40]:40} {tau} {_shown(taus[0]):>6} -> {_shown(taus[1]):>6}"
                f"  {'' if outcome.reranked else 'FELL BACK: ' + str(outcome.reason)}",
                file=out,
            )

    print(
        f"{len(queries)} queries, {comparisons} comparisons asked, {fell_back} fell back to the search's order",
        file=out,
    )
    if seconds:
        print(
            f"Jev: {tokens} input tokens; per query {statistics.median(seconds):.2f}s median, "
            f"{percentile(seconds, 0.9):.2f}s p90, {max(seconds):.2f}s max",
            file=out,
        )
    if llm_input or llm_output:
        print(f"Question LLM: {llm_input} input tokens, {llm_output} output tokens", file=out)
    return report(scores, out)


def report(scores: dict, out=sys.stdout) -> dict:
    """Prints the table of measures for scores = {"search": {measure: [per query]}, "reranked": {...}}, and returns
    the means and the paired differences."""
    summary = {arm: {name: _mean(values) for name, values in measures.items()} for arm, measures in scores.items()}
    names = list(scores["search"])
    summary["difference"] = {name: _paired(scores["search"][name], scores["reranked"][name]) for name in names}
    print(f"{'':20}{'search':>10}{'reranked':>10}{'diff':>9}  {'95% CI':^17}  better/same/worse", file=out)
    for name in names:
        difference = summary["difference"][name]
        interval = f"[{difference['low']:+.3f}, {difference['high']:+.3f}]" if difference["low"] is not None else "n/a"
        print(
            f"{name:20}{_shown(summary['search'][name]):>10}{_shown(summary['reranked'][name]):>10}"
            f"{_signed(difference['mean']):>9}  {interval:^17}  "
            f"{difference['better']}/{difference['same']}/{difference['worse']}",
            file=out,
        )
    return summary


def _paired(search: list[float | None], reranked: list[float | None], resamples: int = 10_000) -> dict:
    """Mean difference over the queries where both are defined, with a 95% paired bootstrap interval (seeded, so a
    re-run on the same answers prints the same interval)."""
    differences = [b - a for a, b in zip(search, reranked, strict=True) if a is not None and b is not None]
    result = {
        "mean": _mean(differences),
        "low": None,
        "high": None,
        "better": sum(d > 1e-9 for d in differences),
        "same": sum(abs(d) <= 1e-9 for d in differences),
        "worse": sum(d < -1e-9 for d in differences),
    }
    if len(differences) >= 2:
        rng = random.Random(0)
        means = sorted(statistics.fmean(rng.choices(differences, k=len(differences))) for _ in range(resamples))
        result["low"], result["high"] = means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]
    return result


def percentile(values: list[float], share: float) -> float:
    """Nearest rank: the smallest value at least `share` of the values are not above."""
    ordered = sorted(values)
    return ordered[max(math.ceil(share * len(ordered)) - 1, 0)]


def _mean(values: list[float | None]) -> float | None:
    defined = [value for value in values if value is not None]
    return statistics.fmean(defined) if defined else None


def _shown(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _signed(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.3f}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("judged", help="JSONL file of judged queries")
    parser.add_argument("--limit", type=int, help="only the first this many queries")
    parser.add_argument("--per-query", action="store_true", help="print each query's tau-b before and after")
    parser.add_argument("--ranker", help="round-robin, blitz or decisions (default: RERANKER_RANKER)")
    parser.add_argument("--depth", type=int, help="measure on the first this many hits (nDCG@depth of all)")
    arguments = parser.parse_args(argv)
    with open(arguments.judged, encoding="utf-8") as file:
        queries = [json.loads(line) for line in file if line.strip()]
    asyncio.run(
        evaluate(
            queries[: arguments.limit],
            Settings(),
            per_query=arguments.per_query,
            ranker=arguments.ranker,
            depth=arguments.depth,
        )
    )


if __name__ == "__main__":
    main()
