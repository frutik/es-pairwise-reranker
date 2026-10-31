"""Does reranking agree with people better than the search's own order? Reranks judged queries with the real judge and
reports nDCG@10, Kendall's tau-b and pairwise accuracy for both orders, as the Quepid experiments did.

    rerank-eval judged.jsonl [--limit 30]

Each line of the input is one query, its hits in the search's order, each with a human grade:

    {"query": "angle bracket", "documents": [{"text": "Simpson Strong-Tie ...", "grade": 3}, ...]}

It calls Jev (RERANKER_JEV_API_KEY and the other settings apply), so it costs what the comparisons cost.
"""

import argparse
import asyncio
import json
import statistics
import sys

import httpx

from reranker.cache import ComparisonCache
from reranker.config import Settings
from reranker.jev import JevClient
from reranker.metrics import kendall_tau_b, ndcg, pairwise_accuracy
from reranker.service import Reranker, rubric

MEASURES = {"nDCG@10": ndcg, "tau-b": kendall_tau_b, "pairwise accuracy": pairwise_accuracy}


async def evaluate(queries: list[dict], settings: Settings, out=sys.stdout) -> dict:
    async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
        jev = JevClient(
            client,
            url=settings.jev_url,
            api_key=settings.jev_api_key,
            model=settings.jev_model,
            instructions=settings.instructions,
            tie_band=settings.tie_band,
            max_document_chars=settings.max_document_chars,
        )
        return await evaluate_with(
            Reranker(jev, ComparisonCache(settings.cache_size, rubric(settings)), settings), queries, out
        )


async def evaluate_with(reranker: Reranker, queries: list[dict], out=sys.stdout) -> dict:
    scores = {"search": {name: [] for name in MEASURES}, "reranked": {name: [] for name in MEASURES}}
    fell_back = comparisons = 0
    for line in queries:
        texts = [document["text"] for document in line["documents"]]
        grades = [float(document["grade"]) for document in line["documents"]]
        outcome = await reranker.rerank(line["query"], texts)
        fell_back += not outcome.reranked
        comparisons += outcome.asked
        reranked_grades = [grades[ranked.index] for ranked in outcome.results]
        for name, measure in MEASURES.items():
            scores["search"][name].append(measure(grades))
            scores["reranked"][name].append(measure(reranked_grades))

    summary = {arm: {name: _mean(values) for name, values in measures.items()} for arm, measures in scores.items()}
    print(
        f"{len(queries)} queries, {comparisons} comparisons asked, {fell_back} fell back to the search's order",
        file=out,
    )
    print(f"{'':20}{'search':>10}{'reranked':>10}", file=out)
    for name in MEASURES:
        print(f"{name:20}{_shown(summary['search'][name]):>10}{_shown(summary['reranked'][name]):>10}", file=out)
    return summary


def _mean(values: list[float | None]) -> float | None:
    defined = [value for value in values if value is not None]
    return statistics.fmean(defined) if defined else None


def _shown(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("judged", help="JSONL file of judged queries")
    parser.add_argument("--limit", type=int, help="only the first this many queries")
    arguments = parser.parse_args(argv)
    with open(arguments.judged, encoding="utf-8") as file:
        queries = [json.loads(line) for line in file if line.strip()]
    asyncio.run(evaluate(queries[: arguments.limit], Settings()))


if __name__ == "__main__":
    main()
