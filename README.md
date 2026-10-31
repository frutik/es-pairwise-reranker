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

So, in Quepid: **one clear win, one inconclusive result.** This service, run on all of WANDS' usable queries, then
gave a clear win there too (see [Results](#results)). Measure it on your own data before trusting it with traffic
(see [Evaluating](#evaluating)).

Other findings from the experiments that shaped this service:

- **Position bias** depends on the documents. On long legal texts Jev favoured the document shown first 78% of the
  time; on short product records about 55%. Half the pairs are therefore shown the other way round, chosen by a hash
  of the pair so an answer can be cached and repeated.
- **Round-robin** (compare every pair, count wins, a tie is half) copes with a judge that contradicts itself (A beats B
  beats C beats A), which happened 82 times in 991 comparisons. Cheaper methods (sorting with the judge as the
  comparator, or BlitzRank) were not tested.
- Model: `jev-latest` (1.14.0 at the time). The judging instructions are in `reranker/config.py`.

## Results

**Pairwise reranking clearly beats BM25 on WANDS.** Over all 222 usable queries it improves nDCG@10, tau-b and
pairwise accuracy, with every 95% interval well above zero. It adds about 0.6 s to a new query and about 50 ms to a
repeated one.

This service, run through Elasticsearch on WANDS, measured on 2026-10-31.

**Setup.**
- **Stack:** the compose stack, Elasticsearch 9.5.5, with the reranker's default settings (8 s time budget, 16
  comparisons in parallel, Redis cache). The judge is real Jev (`jev-latest`).
- **Baseline:** BM25, a `multi_match` on product name (boosted 3×), product class, category path and description.
- **Reranked:** the same search through the `linear` retriever of
  [`search_example.json`](elasticsearch/search_example.json). The top 10 hits are reranked on `rerank_text`.
- **Queries:** every WANDS query whose BM25 top 10 is fully labelled and not all graded the same, 222 of 480. The
  others can't be scored (an unlabelled hit) or score the same in any order.
- **Grades:** WANDS' labels, Exact 2, Partial 1, Irrelevant 0.

**Agreement with the human labels** (95% paired bootstrap interval on the per-query difference):

| Measure | BM25 | Pairwise | Difference | 95% interval | Queries better / same / worse |
| --- | --- | --- | --- | --- | --- |
| nDCG@10 | 0.903 | 0.951 | **+0.048** | +0.032 to +0.065 | 130 / 47 / 45 |
| Kendall's tau-b | 0.168 | 0.396 | **+0.228** | +0.173 to +0.284 | 134 / 50 / 38 |
| Pairwise accuracy | 0.648 | 0.822 | **+0.174** | +0.129 to +0.219 | 134 / 50 / 38 |

Every interval is well above zero. A first run on a random 30 of these queries pointed the same way, but only tau-b
cleared the noise (+0.151, interval +0.005 to +0.293), so 30 queries were too few for this dataset. Looking at the
two worst queries of that run, the losses came mostly from doubtful labels. For "desk for kids", "kids desk" is
labelled Exact and every other "<brand> kids desk" Irrelevant.

**Search time, and the delay reranking adds** (per query, as the Elasticsearch client sees it):

| | p50 | p90 | p95 | p99 | max |
| --- | --- | --- | --- | --- | --- |
| Plain BM25 search | 5 ms | 7 ms | 7 ms | 11 ms | 25 ms |
| Reranked, cold cache | 618 ms | 687 ms | 705 ms | 822 ms | 960 ms |
| Reranked, warm cache | 57 ms | 61 ms | 62 ms | 64 ms | 64 ms |
| **Added**, cold cache | **+612 ms** | +683 ms | +698 ms | +817 ms | +949 ms |
| **Added**, warm cache | **+53 ms** | +56 ms | +57 ms | +58 ms | +60 ms |

"Cold" is the first time a query is asked, with all 45 comparisons new. "Warm" is the same search again, with every
comparison in the cache. A changed result list falls in between: only its new pairs are asked (see
[What is cached](#what-is-cached)). No query fell back to the search's order; the slowest stayed far inside the 8 s
budget.

**Cost:** 45 comparisons per new query, 9,990 in all. In-process, Jev reported about 625 input tokens per
comparison. That is about 28,000 per new query and about 6.2 million for the run. (Through Elasticsearch the counts
are not passed back, so these come from the in-process run.)

**Limits.**
- One dataset and one baseline query. A stronger first-stage search would leave less to gain.
- WANDS' labels are noisy in places.
- The 222 queries exclude those that could not be scored.
- Pairwise has not yet been compared with pointwise Jev on this data. In Quepid's experiments, on other data,
  pointwise came between the search and pairwise.

### Catastrophy: Reranking a window of 30

**Better results, at eight times the delay.** Reranking BM25's top 30 instead of its top 10 more than doubles the
gain in nDCG@10, because good hits from below position 10 are brought up. But a new query waits about 4.9 s instead
of 0.6 s, too long for a live search.

The setup is the same as above, except:
- **Window:** the reranker gets BM25's top **30** hits, and the measures are taken on the top **10** of each order.
  nDCG@10 is measured against the ideal top 10 of all 30 graded hits, so pulling a good hit up from below position 10
  counts. tau-b@10 and pairwise accuracy@10 score the order of the 10 hits shown.
- **Queries:** the 204 whose top 30 are all labelled and not all graded the same.
- **Settings:** `RERANKER_MAX_DOCUMENTS=30`, and a 60 s time budget so that no query would fall back. None needed
  it: the slowest took 5.2 s, inside the default 8 s.

**Agreement with the human labels:**

| Measure (top 10) | BM25 | Pairwise | Difference | 95% interval | Queries better / same / worse |
| --- | --- | --- | --- | --- | --- |
| nDCG@10 | 0.808 | 0.921 | **+0.113** | +0.089 to +0.138 | 121 / 54 / 29 |
| tau-b@10 | 0.173 | 0.323 | **+0.099** | +0.036 to +0.165 | 64 / 29 / 35 |
| Pairwise accuracy@10 | 0.651 | 0.778 | **+0.089** | +0.036 to +0.144 | 64 / 31 / 33 |

- **nDCG@10** gains +0.113, against +0.048 with a window of 10. BM25 starts lower here (0.808, not 0.903) because
  the ideal is now the best 10 of 30, and reranking 30 recovers most of the gap.
- **tau-b@10 and pairwise accuracy@10** only count queries whose top 10 is not all one grade. Many top 10s became all
  one grade after reranking (66, against 36 for BM25), and those have no order left to score, so 128 queries count. So the
  "Difference" column (per query, both defined) is not the difference of the two means.

**Search time, and the delay reranking adds** (per query, as the Elasticsearch client sees it):

| | p50 | p90 | p95 | p99 | max |
| --- | --- | --- | --- | --- | --- |
| Plain BM25 search (30 hits) | 6 ms | 8 ms | 9 ms | 11 ms | 30 ms |
| Reranked, cold cache | 4,903 ms | 5,033 ms | 5,057 ms | 5,086 ms | 5,217 ms |
| Reranked, warm cache | 57 ms | 61 ms | 62 ms | 66 ms | 79 ms |
| **Added**, cold cache | **+4,898 ms** | +5,029 ms | +5,050 ms | +5,080 ms | +5,206 ms |
| **Added**, warm cache | **+51 ms** | +55 ms | +56 ms | +60 ms | +72 ms |

**Why so slow:** 30 hits are 435 comparisons per query, against 45. At 16 in parallel, that is about 27 rounds of Jev
calls instead of 3, and the delay grows with them: 0.6 s to 4.9 s. A repeated query is as fast as before (+51 ms),
because every comparison comes from the cache.

**Cost:** 435 comparisons per new query, 88,740 in all. At about 625 input tokens per comparison (measured in-process
on WANDS), that is about 272,000 tokens per new query, 10× the window of 10, and about 55 million for the run.

**What it is good for as it is:** work that can wait, such as offline reranking, precomputing the most frequent
queries, or warming the cache in advance. For live search, the number of rounds has to come down. One way is
batching a group's pairs into one Jev request; another is a BlitzRank-style schedule that asks only what can change
the top 10 (see [What to try next](#what-to-try-next)).

To reproduce:

```sh
docker compose up -d --build && docker compose wait setup
docker compose --profile wands run --rm wands
.venv/bin/wands-judged judged-wands-all.jsonl
docker compose exec redis redis-cli FLUSHDB
.venv/bin/es-eval judged-wands-all.jsonl --per-query
```

## How it works

1. Elasticsearch sends the query and the text of the top `rank_window_size` hits.
2. The first `max_documents` (default 20) are compared in every pair: `n(n-1)/2` comparisons, 45 for 10 hits and 190
   for 20. Each comparison is one Jev request, as in the experiments, sent in parallel (`concurrency`).
3. Each answer is cached per query and pair of texts, so a repeated query costs nothing and a changed result list
   only pays for its new pairs ([What is cached](#what-is-cached)).
4. Each hit's score is the share of the others it beat. Hits past `max_documents` all get one score, below the
   reranked ones (`meta.beyond_window` counts them).
5. **Fail open:** if Jev errors or the `time_budget_seconds` runs out, every hit gets the same score, so a slow or
   failing judge never breaks search. The response's `meta.reranked` says which happened.
6. **Equal scores mean "no opinion"**, and the caller breaks them with the search's order. That order cannot be
   decided here: Elasticsearch sends the hits in index order, not score order, and without their scores. In
   Elasticsearch that is what the `linear` retriever in the [example search](#wiring-it-into-elasticsearch) does. The
   results list itself is sorted by score and then by input position, which is right for callers that send hits in
   the search's order (`rerank-eval`).

## What is cached

One entry is **one comparison**: for a query and two document texts A and B, which of the two Jev preferred. The
cache holds neither result lists nor scores.

| | |
| --- | --- |
| **Key** | SHA-256 of: the judge (the rubric, below), the query text, and the two document texts in a fixed order (the smaller string first), so "A vs B" and "B vs A" are one entry. In Redis: `reranker:comparison:<hex>` |
| **Value** | The winner, `first`, `second` or `tie`, relative to that fixed order. Read back the other way round, it is flipped. Jev's probability and confidence are not kept |
| **Rubric** (the judge) | Jev URL, model name, judging instructions, tie band and the truncation length (`RERANKER_MAX_DOCUMENT_CHARS`). Change any of them and every comparison is asked again, because it is a different question |

**Not in the key:** the hits' positions or BM25 scores, document ids, the other hits of the result list, and the
time. The texts are hashed exactly as Elasticsearch sends them (the reranker's `field`): no lowercasing, trimming or
other normalising, of the query or of the documents. Neither texts nor queries are stored, only their hash, so Redis
holds no document content.

What that means when a query's results change (10 hits, 45 pairs):

| What changed | Comparisons asked again |
| --- | --- |
| Nothing: the same query and hits | none |
| The same hits, in a different BM25 order | none: the answers do not depend on position |
| One new hit in the top 10 | 9: its pairs with the other nine |
| One document's text edited | 9: the edited text is a new text |
| The query typed differently ("Salon chair", "salon chair ") | all 45 |
| Another query, the same hits | all 45: the judgement depends on the query |

An answer is never served for a text it was not asked about. Entries for texts that are no longer around are just
not read again, until they are evicted.

**Watch the model name.** `jev-latest` is an alias. If TypeSafe moves it to a new model, the key does not change, and
the old model's cached answers are mixed with the new one's. Pin a version in `RERANKER_JEV_MODEL` to make an upgrade
a new key, or set `RERANKER_CACHE_TTL_SECONDS` so that old answers expire.

**Where it is kept:**

- **Without `RERANKER_REDIS_URL`:** in each process's memory, `RERANKER_CACHE_SIZE` entries (100,000), least
  recently used first out. Several workers or replicas each keep and pay for their own.
- **With it:** in Redis, shared by every worker and replica. Redis's `maxmemory` policy evicts (the compose stack
  runs `allkeys-lru` at 256 MB), and `RERANKER_CACHE_TTL_SECONDS`, when set, expires each entry.
- **If Redis is down,** or slower than `RERANKER_REDIS_TIMEOUT_SECONDS` (0.5), its answers count as misses: the
  comparisons are asked again and search keeps working.

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

If you recreate the reranker container (`up --force-recreate reranker`, or a changed setting), restart
Elasticsearch too (`docker compose restart elasticsearch`, then `docker compose run --rm setup`). Elasticsearch
keeps the old container's address and its rerank calls fail with "Connection refused".

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

**Cache.** Set `RERANKER_REDIS_URL` to share comparisons between workers and replicas (the compose stack does); see
[What is cached](#what-is-cached).

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

**Through Elasticsearch, with the delay it adds.** `es-eval` runs each judged query twice through the stack: as the
plain search, and reranked (the `linear` retriever of the example). It scores both orders against the grades, as
above. It also reports search time and the delay reranking adds, at p50, p90, p95, p99 and max. Delay is measured
twice:
- **cold**: the first reranked search of each query, when every comparison is new;
- **warm**: the same searches again, when every comparison comes from the cache, with no Jev cost.

It uses the stack's own settings, including the 8s time budget, so it measures what live traffic would see.

```sh
docker compose exec redis redis-cli FLUSHDB      # empty the cache, so "cold" is cold
.venv/bin/es-eval judged-wands-all.jsonl --per-query    # judged-wands-all.jsonl: wands-judged without --sample
```

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
