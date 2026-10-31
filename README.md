# es-pairwise-reranker

An Elasticsearch reranker that orders search hits by asking a model **which of two documents is more relevant to the
query**, for every pair of the top hits, and ranking them by how many comparisons each won.

It is a small HTTP service. Elasticsearch (8.19+ / 9.1+) calls it through a `custom` inference endpoint from a
`text_similarity_reranker` retriever, so reranking is part of an ordinary search request.

The judge is [TypeSafe Jev](https://typesafe.ai), a typed evaluation model: it is given a state (the query and two
documents) and a typed question (a choice between "A is more relevant" and "B is more relevant"), and answers with
the choice, a probability for it and a confidence. It writes no prose.

## Why pairwise

Rating one document at a time ("is this a 2 or a 3?") is a hard question for a model. "Which of these two is better?"
is easier, for models as for people. An order built from enough of those answers can be better than the order implied
by grading each document. An order is not a set of grades, though: it says A beats B, not whether either is any good.
So this service only reorders. Its scores are for sorting one query's hits, not for comparing across queries.

## What to try next

Comparing every pair is the most expensive way to get an order from pairwise answers: `n(n-1)/2` comparisons, so
190 for 20 hits. It is what the experiments measured, so it is what the service does. These are the ways to make it
cheaper or better, roughly in order of promise. Each one has to be measured against the current version (see
[Evaluating](#evaluating)) before it replaces it.

- **BlitzRank** ([Agrawal, Nguyen and Kiela, ICML 2026](https://arxiv.org/abs/2602.05448);
  [code](https://github.com/ContextualAI/BlitzRank), `pip install blitzrank`). It asks the judge to order a small
  set of `k` documents at once, which gives every pairwise preference within that set in one call. It collects these
  into one tournament graph and uses transitivity (A beats B, B beats C, so A beats C) to infer orderings it never
  asked about. The paper reports quality close to full pairwise ranking with about 7× fewer tokens, across 14
  benchmarks and 5 models. Two open questions here: Jev would have to answer a `k`-way question instead of an A/B
  one, and the swap against position bias would need a counterpart for orderings of `k` documents.
- **Comparison sort**: a sort that uses the judge as its comparator needs about `n log n` comparisons, not `n²/2`.
  Quepid has one to port (`app/services/pair_ranking/comparison_sort.rb`). Round-robin copes with a judge that
  contradicts itself (82 cycles in 991 comparisons in the experiments); a sort does not, so check how much accuracy
  that costs.
- **Batching**: send a query's pairs to Jev as several questions in one request, with the documents in the state
  once. This leaves the method unchanged and cuts requests and latency.
- **Richer documents**: rerank on a combined field (title, brand, description) rather than the title alone. The
  experiments showed Jev whole records with named fields.

## Evidence so far

This is the method from experiments run in [Quepid](https://github.com/o19s/quepid) (a relevance-testing tool) on
human-rated data. Each query's rated documents were ordered three ways and scored against the human grades:

**Haystack Icecat rating party** (30 queries, 254 query-document pairs, Icecat product records, crowd ratings 0–3
averaged per pair):

| Ordering | Kendall's tau-b | nDCG@10 | Pairwise accuracy |
| --- | --- | --- | --- |
| The search's own order | 0.019 | 0.799 | 0.505 |
| Jev grading each document (pointwise) | 0.340 | 0.879 | 0.705 |
| Jev comparing every pair (this method) | **0.466** | **0.922** | **0.804** |

Pairwise beat pointwise by +0.127 tau-b (95% paired-bootstrap interval 0.028 to 0.225) and +0.043 nDCG@10 (0.018 to
0.075), better on 20 of 30 queries. Reversing the candidate order barely changed the result (tau-b 0.893 between the
two). Cost: 991 comparisons, about 670 tokens each.

**WANDS** (Wayfair's public furniture dataset, labels 0–2; 30 queries, the top 10 of a text search): pairwise was ahead
again (tau-b 0.421 against 0.354; nDCG@10 0.979 against 0.972) but within the noise. The search's own top 10 already
scored 0.957, which left little to improve.

So: **one clear win, one inconclusive result.** Measure it on your own data before trusting it with traffic (see
[Evaluating](#evaluating)).

Other findings from the experiments that shaped this service:

- **Position bias** depends on the documents. On long legal texts Jev favoured the document shown first 78% of the
  time; on short product records about 55%. Half the pairs are therefore shown the other way round, chosen by a hash
  of the pair so an answer can be cached and repeated.
- **Round-robin** (compare every pair, count wins, a tie is half) copes with a judge that contradicts itself (A beats B
  beats C beats A), which happened 82 times in 991 comparisons. Cheaper methods (sorting with the judge as the
  comparator, or BlitzRank) were not tested.
- Model: `jev-latest` (1.14.0 at the time). The judging instructions are in `reranker/config.py`.

## How it works

1. Elasticsearch sends the query and the text of the top `rank_window_size` hits.
2. The first `max_documents` (default 20) are compared in every pair: `n(n-1)/2` comparisons, 45 for 10 hits and 190
   for 20. Each comparison is one Jev request, as in the experiments, sent in parallel (`concurrency`).
3. Answers are cached in memory per model, instructions, query and pair, so a repeated query costs nothing.
4. Each hit's score is the share of the others it beat. Hits past `max_documents` all get one score, below the
   reranked ones (`meta.beyond_window` counts them).
5. **Fail open:** if Jev errors or the `time_budget_seconds` runs out, every hit gets the same score, so a slow or
   failing judge never breaks search. The response's `meta.reranked` says which happened.
6. **Equal scores mean "no opinion"**, and the caller breaks them with the search's order. That order cannot be
   decided here: Elasticsearch sends the hits in index order, not score order, and without their scores. In
   Elasticsearch that is what the `linear` retriever in the [example search](#wiring-it-into-elasticsearch) does. The
   results list itself is sorted by score and then by input position, which is right for callers that send hits in
   the search's order (`rerank-eval`).

## Running it

```sh
python3.13 -m venv .venv   # 3.12+ && .venv/bin/pip install -e '.[dev]'
cp .env.example .env   # set RERANKER_JEV_API_KEY, and RERANKER_SERVICE_API_KEY
.venv/bin/uvicorn reranker.app:app --port 8000
```

Or with Docker: `docker build -t es-pairwise-reranker . && docker run --env-file .env -p 8000:8000 es-pairwise-reranker`.

```sh
curl -s localhost:8000/v1/rerank -H 'Authorization: Bearer <service key>' -H 'Content-Type: application/json' \
  -d '{"query": "angle bracket", "documents": ["Patio chair", "Simpson Strong-Tie 12-Gauge Angle", "Shelf bracket"]}'
# {"results": [{"index": 1, "relevance_score": 1.0}, ...], "meta": {"reranked": true, "comparisons": 3, ...}}
```

All settings are environment variables, listed with their defaults in `.env.example` and `reranker/config.py`.

**The whole stack** (Elasticsearch 9.5, the reranker with two workers, Redis as their shared cache) runs with Docker
Compose. A one-off `setup` container registers the `pairwise-jev` inference endpoint once Elasticsearch and the
reranker are healthy:

```sh
cp .env.example .env            # set RERANKER_JEV_API_KEY; without it every rerank falls back to the search's order
docker compose up -d --build
docker compose wait setup       # `up -d` returns before the endpoint is registered; this waits for it (exit code 0)
curl -s localhost:9200/_inference/rerank/pairwise-jev
```

**Without a Jev key**, or to avoid paying for comparisons, run the stack against a fake Jev
([`fake_jev/server.py`](fake_jev/server.py)). It prefers the document that shares more words with the query, so it
checks the wiring, the cache, timeouts and fail-open; it says nothing about relevance quality.

```sh
RERANKER_JEV_URL=http://fake-jev:9000 docker compose --profile fake-jev up -d --build
FAKE_JEV_FAIL=1 RERANKER_JEV_URL=http://fake-jev:9000 docker compose --profile fake-jev up -d   # every Jev call 503s
FAKE_JEV_DELAY_SECONDS=10 ...                                                                  # every Jev call is slow
```

The judge's URL is part of the cache key, so the fake's answers never stand in for real Jev's in Redis.

It is for local use only: Elasticsearch security is off (on a 30-day trial licence) and the ports bind to 127.0.0.1.
If the default ports are taken, change them with `ES_PORT`, `RERANKER_PORT` and `REDIS_PORT`, e.g.
`ES_PORT=9201 docker compose up -d`. `WEB_CONCURRENCY` sets the number of reranker workers.

**Trying it on WANDS.** [WANDS](https://github.com/wayfair/WANDS) is Wayfair's public furniture search dataset
(42,994 products, MIT licence). To load it into the stack's Elasticsearch as the index `wands`:

```sh
docker compose --profile wands run --rm wands     # downloads it from GitHub the first time, then indexes it
# or, without Docker for the loader: .venv/bin/wands-load --es http://localhost:9200   (files go to data/wands/)
```

Re-running replaces the index. The files are downloaded once: into the `wands-data` volume, or `--data-dir` on the
host. Each product keeps WANDS' fields and adds `rerank_text`, which holds the name, category path, product class and
the first 1,000 characters of the description. Point the reranker at that field, since Elasticsearch sends the judge
only one field:

```sh
Q='{"multi_match": {"query": "salon chair",
     "fields": ["product_name^3", "product_class", "category_hierarchy", "product_description"]}}'
curl -s localhost:9200/wands/_search -H 'Content-Type: application/json' -d '{
  "retriever": {"linear": {"rank_window_size": 10, "retrievers": [
    {"retriever": {"text_similarity_reranker": {"retriever": {"standard": {"query": '"$Q"'}},
       "field": "rerank_text", "inference_id": "pairwise-jev", "inference_text": "salon chair",
       "rank_window_size": 10}}, "weight": 1},
    {"retriever": {"standard": {"query": '"$Q"'}}, "weight": 0.001, "normalizer": "minmax"}]}}}'
```

**Cache.** Without `RERANKER_REDIS_URL` each process keeps its own in-memory cache. With it, every worker and replica
shares one cache in Redis (run Redis with a `maxmemory` policy such as `allkeys-lru`). If Redis is down or slower
than `RERANKER_REDIS_TIMEOUT_SECONDS`, its answers count as cache misses: the comparisons are asked again, and search
keeps working.

## Wiring it into Elasticsearch

Create the inference endpoint (8.19+ / 9.1+), with the service's URL and key in
[`elasticsearch/inference_endpoint.json`](elasticsearch/inference_endpoint.json):

```sh
curl -X PUT "$ES/_inference/rerank/pairwise-jev" -H 'Content-Type: application/json' \
  -d @elasticsearch/inference_endpoint.json
```

It uses Elasticsearch's `custom` service: `${query}` and `${input}` (the hits' texts) fill the request, and the
response's `results[*].relevance_score` and `results[*].index` are read back.

Then rerank in a search ([`elasticsearch/search_example.json`](elasticsearch/search_example.json)):

```json
{
  "retriever": {
    "linear": {
      "rank_window_size": 10,
      "retrievers": [
        {
          "retriever": {
            "text_similarity_reranker": {
              "retriever": { "standard": { "query": { "match": { "title": "angle bracket" } } } },
              "field": "title",
              "inference_id": "pairwise-jev",
              "inference_text": "angle bracket",
              "rank_window_size": 10
            }
          },
          "weight": 1
        },
        {
          "retriever": { "standard": { "query": { "match": { "title": "angle bracket" } } } },
          "weight": 0.001,
          "normalizer": "minmax"
        }
      ]
    }
  }
}
```

**Why `linear` around the reranker.** Elasticsearch sends the reranker the top hits in index order, without their
scores, and orders equal rerank scores by index too. A plain `text_similarity_reranker` therefore shows ties, and the
whole fallback, in index order. Adding the original query again with a tiny weight breaks those ties by its own
score, so a tie or a fallback comes back in the search's order. 0.001 has to stay below the smallest gap between
rerank scores, half a win out of `n - 1` (0.026 for 20 hits). The query runs twice; that costs much less than the
comparisons.

Things to know:

- **Only `field` is sent.** The judge sees that one field's text. If titles alone are not enough, index a combined
  field (title, brand, description) and rerank on that.
- **`rank_window_size` is the cost.** 10 hits are 45 comparisons. Keep it at or below `max_documents`. Beyond that,
  the hits that get reranked are the first in index order, not the best (the response's `meta.beyond_window` and
  a warning in the log say so).
- **A failing endpoint fails the search.** If the reranker answers with an error, Elasticsearch fails the whole
  search (400). That is why the service always answers 200 and falls back with equal scores instead.
- **Timeouts.** Keep the service's `time_budget_seconds` below Elasticsearch's inference timeout, so a slow judge
  falls back to the search's order instead of failing the search.

## Evaluating

Before sending traffic through it, check that it beats your current order on your own judged queries.

**From a file** of judged queries, one JSON object a line, hits in the search's order:

```json
{"query": "angle bracket", "documents": [{"text": "Simpson Strong-Tie 12-Gauge Angle", "grade": 3}, {"text": "Patio chair", "grade": 0}]}
```

```sh
.venv/bin/rerank-eval judged.jsonl --limit 30
# 30 queries, <n> comparisons asked, <n> fell back to the search's order
#                         search  reranked
# nDCG@10                  0.xxx     0.xxx
# tau-b                    0.xxx     0.xxx
# pairwise accuracy        0.xxx     0.xxx
```

It calls Jev for real, so it costs what the comparisons cost. Besides the means it prints, per measure, the mean
difference, a 95% paired bootstrap interval for it, and on how many queries reranking was better, the same or worse.
`--per-query` prints each query's tau-b before and after.

**On WANDS**, with the `wands` index loaded (see [Running it](#running-it)):

```sh
.venv/bin/wands-judged judged-wands.jsonl --sample 30 --seed 0   # BM25 top 10 per query, graded with WANDS' labels
RERANKER_TIME_BUDGET_SECONDS=120 .venv/bin/rerank-eval judged-wands.jsonl --per-query
```

`wands-judged` keeps only queries whose top hits are all labelled and not all graded the same (222 of 480), so no
grade is guessed, and samples from them with a fixed seed. Raise the time budget for evaluation, so that a slow Jev
shows up as latency rather than as fallbacks.

**In Quepid:** make two search endpoints for the same index, one with the `text_similarity_reranker` retriever and
one without, and compare them on a rated case. Elasticsearch does the reranking, so nothing else is needed.

## Not done yet

- **Batching.** Jev can answer several questions in one request with the documents sent once, which would cut
  requests and latency a lot. It was not measured in the experiments, so it is not used yet.
- **Cheaper rankers.** BlitzRank or comparison sort, for windows larger than about 20; see
  [What to try next](#what-to-try-next).
- **Checked on one Elasticsearch version only**: 9.5.5, the compose stack. The endpoint is accepted, the reranked
  order is applied, and the `linear` retriever breaks ties and the fallback in BM25 order. Elasticsearch shifts the
  returned scores (+1, and negative ones into (0, 1)) but keeps their order. 8.19 has not been tried.

## Development

```sh
.venv/bin/pytest          # tests, with Jev answered by a fake (no network)
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```
