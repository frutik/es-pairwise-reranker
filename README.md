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
4. Each hit's score is the share of the others it beat. Equal scores keep the search's order. Hits past
   `max_documents` keep their order, below the reranked ones.
5. **Fail open:** if Jev errors or the `time_budget_seconds` runs out, the search's own order is returned, so a slow
   or failing judge never breaks search. The response's `meta.reranked` says which happened.

## Running it

```sh
python3.13 -m venv .venv && .venv/bin/pip install -e '.[dev]'
cp .env.example .env   # set RERANKER_JEV_API_KEY, and RERANKER_SERVICE_API_KEY
.venv/bin/uvicorn reranker.app:app --port 8000
```

Or with Docker: `docker build -t es-pairwise-reranker . && docker run --env-file .env -p 8000:8000 es-pairwise-reranker`.

**The whole stack** (Elasticsearch 9.5, the reranker with two workers, Redis as their shared cache) runs with Docker
Compose. A one-off `setup` container registers the `pairwise-jev` inference endpoint once Elasticsearch and the
reranker are healthy:

```sh
cp .env.example .env            # set RERANKER_JEV_API_KEY; without it every rerank falls back to the search's order
docker compose up -d --build
curl -s localhost:9200/_inference/rerank/pairwise-jev
```

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
curl -s localhost:9200/wands/_search -H 'Content-Type: application/json' -d '{
  "retriever": {"text_similarity_reranker": {
    "retriever": {"standard": {"query": {"multi_match": {"query": "salon chair",
      "fields": ["product_name^3", "product_class", "category_hierarchy", "product_description"]}}}},
    "field": "rerank_text", "inference_id": "pairwise-jev", "inference_text": "salon chair", "rank_window_size": 10}}}'
```

**Cache.** Without `RERANKER_REDIS_URL` each process keeps its own in-memory cache. With it, every worker and replica
shares one cache in Redis (run Redis with a `maxmemory` policy such as `allkeys-lru`). If Redis is down or slower
than `RERANKER_REDIS_TIMEOUT_SECONDS`, its answers count as cache misses: the comparisons are asked again, and search
keeps working.

```sh
curl -s localhost:8000/v1/rerank -H 'Authorization: Bearer <service key>' -H 'Content-Type: application/json' \
  -d '{"query": "angle bracket", "documents": ["Patio chair", "Simpson Strong-Tie 12-Gauge Angle", "Shelf bracket"]}'
# {"results": [{"index": 1, "relevance_score": 1.0}, ...], "meta": {"reranked": true, "comparisons": 3, ...}}
```

All settings are environment variables, listed with their defaults in `.env.example` and `reranker/config.py`.

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
    "text_similarity_reranker": {
      "retriever": { "standard": { "query": { "match": { "title": "angle bracket" } } } },
      "field": "title",
      "inference_id": "pairwise-jev",
      "inference_text": "angle bracket",
      "rank_window_size": 10
    }
  }
}
```

Things to know:

- **Only `field` is sent.** The judge sees that one field's text. If titles alone are not enough, index a combined
  field (title, brand, description) and rerank on that.
- **`rank_window_size` is the cost.** 10 hits are 45 comparisons; keep it at or below `max_documents`.
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

It calls Jev for real, so it costs what the comparisons cost.

**In Quepid:** make two search endpoints for the same index, one with the `text_similarity_reranker` retriever and
one without, and compare them on a rated case. Elasticsearch does the reranking, so nothing else is needed.

## Not done yet

- **Batching.** Jev can answer several questions in one request with the documents sent once, which would cut
  requests and latency a lot. It was not measured in the experiments, so it is not used yet.
- **Cheaper rankers.** BlitzRank or comparison sort, for windows larger than about 20; see
  [What to try next](#what-to-try-next).
- **The order Elasticsearch sends hits in.** Checked against Elasticsearch 9.5.5 (the compose stack): the endpoint is
  accepted and its order is applied. Elasticsearch shifts the returned scores (+1, and negative scores into (0, 1)),
  but keeps their order. However, it sends the top `rank_window_size` hits in **index order, not score order**. The
  service assumes the search's order in three places: the fallback order, the tiebreak between equal win counts, and
  which hits fall inside `max_documents`. Until that is fixed, keep `rank_window_size` ≤ `max_documents`, and expect
  the fallback to return index order.

## Development

```sh
.venv/bin/pytest          # tests, with Jev answered by a fake (no network)
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```
