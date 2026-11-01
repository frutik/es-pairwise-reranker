"""Reranking measured where it runs: through Elasticsearch. For each judged query it times the plain search and the
reranked one (a `linear` retriever around the text_similarity_reranker, as in elasticsearch/search_example.json),
scores both orders against the grades, and reports the delay reranking adds.

    es-eval judged-wands.jsonl [--es http://localhost:9200] [--index wands] [--inference-id pairwise-jev]
                               [--field rerank_text] [--window 10] [--depth 10] [--limit N] [--per-query]

The judged file is rerank-eval's, with each document's "id" (wands-judged writes it). The plain search is the same
multi_match wands-judged ran, so its order is the one the grades were taken from.

Delay is measured twice:
- cold: the first reranked search of each query. Every comparison is new, so this is what a query nobody has asked
  before waits. Empty the comparison cache before the run (redis-cli FLUSHDB) for this to be true.
- warm: the same searches again, every comparison now in the cache. No Jev calls, so no cost.

Each query's added delay is its reranked search's time minus its plain search's, as Elasticsearch's client sees it
(HTTP included). The plain search is run once untimed first, so both find Elasticsearch equally warm.

--window hits are reranked; the measures are taken on the first --depth of each order (by default all of them). With
--window 30 --depth 10, nDCG@10 is measured against the ideal order of all 30 graded hits, so pulling a good hit up
from below the top 10 counts. The judged file has to grade all --window hits (wands-judged --size 30).

It costs what the cold comparisons cost: one Jev request per pair of each query's top --window hits.
"""

import argparse
import json
import os
import sys
import time

import httpx

from eval.evaluate import MEASURES, measures_at, percentile, report
from eval.wands import SEARCH_FIELDS

# BM25's weight next to the reranker's: below the smallest gap between rerank scores, half a win out of n - 1
# (see the README, "Why linear around the reranker").
TIEBREAK_WEIGHT = 0.001


def plain_search(text: str, window: int) -> dict:
    return {"size": window, "_source": False, "query": _query(text)}


def reranked_search(text: str, window: int, inference_id: str, field: str) -> dict:
    standard = {"standard": {"query": _query(text)}}
    return {
        "size": window,
        "_source": False,
        "retriever": {
            "linear": {
                "rank_window_size": window,
                "retrievers": [
                    {
                        "retriever": {
                            "text_similarity_reranker": {
                                "retriever": standard,
                                "field": field,
                                "inference_id": inference_id,
                                "inference_text": text,
                                "rank_window_size": window,
                            }
                        },
                        "weight": 1,
                    },
                    {"retriever": standard, "weight": TIEBREAK_WEIGHT, "normalizer": "minmax"},
                ],
            }
        },
    }


def _query(text: str) -> dict:
    return {"multi_match": {"query": text, "fields": SEARCH_FIELDS}}


def timed_search(client: httpx.Client, index: str, body: dict) -> tuple[list[dict], float]:
    started = time.perf_counter()
    response = client.post(f"/{index}/_search", json=body)
    seconds = time.perf_counter() - started
    if response.status_code != 200:
        raise RuntimeError(f"search failed: {response.status_code} {response.text[:500]}")
    return response.json()["hits"]["hits"], seconds


def delay_line(name: str, values: list[float]) -> str:
    points = [("p50", 0.5), ("p90", 0.9), ("p95", 0.95), ("p99", 0.99)]
    shown = "  ".join(f"{label} {percentile(values, share) * 1000:7.0f}" for label, share in points)
    return f"{name:28}{shown}  max {max(values) * 1000:7.0f}  ms"


def run(arguments, queries: list[dict], out=sys.stdout) -> dict:
    measures = MEASURES if arguments.depth is None else measures_at(arguments.depth)
    tau = next(name for name in measures if name.startswith("tau-b"))
    scores = {"search": {name: [] for name in measures}, "reranked": {name: [] for name in measures}}
    plain_seconds, cold_seconds, warm_seconds = [], [], []
    flat = mismatched = 0
    with httpx.Client(base_url=arguments.es, timeout=arguments.timeout) as client:
        for line in queries:
            grade = {document["id"]: float(document["grade"]) for document in line["documents"]}
            text = line["query"]
            timed_search(client, arguments.index, plain_search(text, arguments.window))  # warm-up, not timed
            plain, plain_time = timed_search(client, arguments.index, plain_search(text, arguments.window))
            reranked, cold_time = timed_search(
                client,
                arguments.index,
                reranked_search(text, arguments.window, arguments.inference_id, arguments.field),
            )
            plain_seconds.append(plain_time)
            cold_seconds.append(cold_time)

            plain_ids, reranked_ids = [hit["_id"] for hit in plain], [hit["_id"] for hit in reranked]
            if set(plain_ids) != set(grade) or set(reranked_ids) != set(grade):
                mismatched += 1  # the index changed since the grades were taken: those hits have no grade
            # All scores within the tiebreak's reach of each other: the reranker had no opinion (a fallback, or
            # every comparison a tie).
            if reranked and max(h["_score"] for h in reranked) - min(h["_score"] for h in reranked) <= TIEBREAK_WEIGHT:
                flat += 1
            search_grades = [grade.get(i, 0.0) for i in plain_ids]
            reranked_grades = [grade.get(i, 0.0) for i in reranked_ids]
            for name, measure in measures.items():
                scores["search"][name].append(measure(search_grades))
                scores["reranked"][name].append(measure(reranked_grades))
            if arguments.per_query:
                print(
                    f"{text[:40]:40} plain {plain_time * 1000:5.0f} ms  reranked {cold_time * 1000:6.0f} ms"
                    f"  {tau} {_shown(scores['search'][tau][-1])} -> {_shown(scores['reranked'][tau][-1])}",
                    file=out,
                )

        for line in queries:
            _, warm_time = timed_search(
                client,
                arguments.index,
                reranked_search(line["query"], arguments.window, arguments.inference_id, arguments.field),
            )
            warm_seconds.append(warm_time)

    print(
        f"\n{len(queries)} queries through {arguments.es}/{arguments.index}, top {arguments.window} reranked,"
        f" measured on the top {arguments.depth or arguments.window}",
        file=out,
    )
    if mismatched:
        print(f"{mismatched} queries returned hits the judged file has no grade for (counted as 0)", file=out)
    print(f"{flat} queries came back with no rerank opinion (fell back, or all ties)", file=out)
    print("\nSearch time", file=out)
    print(delay_line("plain", plain_seconds), file=out)
    print(delay_line("reranked, cold cache", cold_seconds), file=out)
    print(delay_line("reranked, warm cache", warm_seconds), file=out)
    print("Added by reranking (reranked - plain, per query)", file=out)
    cold_added = [cold - plain for cold, plain in zip(cold_seconds, plain_seconds, strict=True)]
    warm_added = [warm - plain for warm, plain in zip(warm_seconds, plain_seconds, strict=True)]
    print(delay_line("cold cache", cold_added), file=out)
    print(delay_line("warm cache", warm_added), file=out)
    print("\nAgreement with the grades", file=out)
    summary = report(scores, out)
    summary["delay"] = {"plain": plain_seconds, "cold": cold_seconds, "warm": warm_seconds}
    if arguments.save:
        with open(arguments.save, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "queries": [line["query"] for line in queries],
                    "scores": scores,
                    "delay": summary["delay"],
                    "inference_id": arguments.inference_id,
                },
                file,
            )
    return summary


def _shown(value: float | None) -> str:
    return "  n/a" if value is None else f"{value:+.2f}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("judged", help="JSONL of judged queries, with document ids (wands-judged writes one)")
    parser.add_argument("--es", default=os.environ.get("ES_URL", "http://localhost:9200"), help="Elasticsearch URL")
    parser.add_argument("--index", default="wands")
    parser.add_argument("--inference-id", default="pairwise-jev")
    parser.add_argument("--field", default="rerank_text", help="the field the reranker sends the judge")
    parser.add_argument("--window", type=int, default=10, help="hits reranked per query (keep it <= max_documents)")
    parser.add_argument("--depth", type=int, help="measure on the first this many hits (default: all --window)")
    parser.add_argument("--timeout", type=float, default=60, help="seconds a search may take")
    parser.add_argument("--limit", type=int, help="only the first this many queries")
    parser.add_argument("--per-query", action="store_true", help="print each query's times and tau-b")
    parser.add_argument("--save", help="write each query's measures and times to this JSON file")
    arguments = parser.parse_args(argv)
    with open(arguments.judged, encoding="utf-8") as file:
        queries = [json.loads(line) for line in file if line.strip()][: arguments.limit]
    try:
        run(arguments, queries)
    except (RuntimeError, httpx.HTTPError) as error:
        sys.exit(f"es-eval: {error}")


if __name__ == "__main__":
    main()
