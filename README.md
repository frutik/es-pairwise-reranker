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

- **BlitzRank: done, and it works** (see [BlitzRank: a window of 30](#blitzrank-a-window-of-30)). On a window of
  30 it gives round-robin's quality with about 4× less delay and 8× fewer tokens. `RERANKER_RANKER=blitz`, or call
  `/v1/rerank/blitz`.
- **Bag of decisions: done** (see [Bag of decisions: a window of 30](#bag-of-decisions-a-window-of-30)). 0.013
  below round-robin in nDCG@10, with the fewest Jev tokens and +410 ms once a query's questions are kept, but a new
  query waits for the LLM. `RERANKER_RANKER=decisions`, or call `/v1/rerank/decisions`.
- **Comparison sort**: a sort that uses the judge as its comparator needs about `n log n` comparisons, not `n²/2`.
  Quepid has one to port (`app/services/pair_ranking/comparison_sort.rb`). Round-robin copes with a judge that
  contradicts itself (82 cycles in 991 comparisons in the experiments); a sort does not, so check how much accuracy
  that costs.
- **Batching round-robin**: send a query's pairs to Jev as several questions in one request, with the documents in
  the state once, as BlitzRank's rounds already do. This leaves round-robin unchanged and cuts requests and latency.
- **A smaller BlitzRank group, or more of them in parallel**: BlitzRank's rounds run one after another, about 0.2 s
  each, so its delay is the number of rounds. Groups of other sizes than 10 were not tried.
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
queries, or warming the cache in advance. For live search, the number of rounds has to come down, which is what
BlitzRank does (next).

### BlitzRank: a window of 30

**Round-robin's quality at a quarter of the delay and an eighth of the tokens.** The same 204 queries, window 30
and top 10 as above, ranked with [BlitzRank](https://arxiv.org/abs/2602.05448) instead of round-robin.

**How it runs here** (`reranker/blitz.py`, ported from the authors' package, `blitzrank` 0.1.0):
- **Rounds:** each round asks Jev about a group of 10 documents, every pair of the group in **one request** (the
  documents sent once, one question per pair). The answers go into one preference graph, and its transitive closure
  infers pairs nobody asked.
- **Ties:** documents that beat each other in a circle count as tied.
- **Stopping:** a document is final once its relation to every other is known, and rounds stop when 10 are final.
- **The next group:** one document from each of the 10 best unfinished groups of tied documents, so every round asks
  only what is still open.
- **The rest:** BlitzRank does not order positions 11–30, so they get equal scores below the top 10, in BM25 order.
- **Differences from the authors' code:**
  - Their judge orders the group (a listwise LLM call). Jev answers each pair of the group, which gives at least as
    much: a full tournament.
  - A pair already answered is not asked again.
  - Documents are labelled D1..D10 in an order fixed by their text, not by how good they look so far.

**Agreement with the human labels:**

| Measure (top 10) | BM25 | Round-robin | BlitzRank | BlitzRank − round-robin | 95% interval |
| --- | --- | --- | --- | --- | --- |
| nDCG@10 | 0.808 | 0.921 | 0.919 | −0.002 | −0.008 to +0.004 |
| tau-b@10 | 0.173 | 0.323 | 0.322 | +0.006 | −0.027 to +0.037 |
| Pairwise accuracy@10 | 0.651 | 0.778 | 0.772 | −0.000 | −0.032 to +0.030 |

No difference between the two. The intervals are narrow and centred on zero, and BlitzRank gave the same nDCG@10 as
round-robin on 124 of 204 queries. Against BM25, BlitzRank gains +0.111 nDCG@10 (+0.087 to +0.137).

**Delay reranking adds** (per query, as the Elasticsearch client sees it):

| Added | p50 | p90 | p95 | p99 | max |
| --- | --- | --- | --- | --- | --- |
| Round-robin, cold cache | +4,898 ms | +5,029 ms | +5,050 ms | +5,080 ms | +5,206 ms |
| **BlitzRank, cold cache** | **+1,133 ms** | +1,296 ms | +1,348 ms | +1,398 ms | +1,697 ms |
| BlitzRank, warm cache | +53 ms | +57 ms | +58 ms | +62 ms | +293 ms |

**Cost per new query** (measured in-process on 20 of the queries):

| | Pairs asked | Jev requests | One after another | Input tokens |
| --- | --- | --- | --- | --- |
| Round-robin | 435 | 435 | about 27 rounds of 16 | about 272,000 |
| **BlitzRank** | **195** (median) | 5 | 5 rounds (6 at most) | **about 33,000** |

All 20 reached a certain top 10. Against round-robin's run, BlitzRank's delay is 4.3× smaller and its tokens 8×
fewer. The paper reports 7× fewer tokens than pairwise ranking with other models.

**What is still slow:** 1.1 s is for a query nobody has asked before. That is closer to live search than 4.9 s, but
still a second.

**To try it:** `RERANKER_RANKER=blitz` makes it the default. Or keep round-robin and register a second endpoint on
`/v1/rerank/blitz`:

```sh
docker compose run --rm -e RERANKER_URL=http://reranker:8000/v1/rerank/blitz -e INFERENCE_ID=pairwise-blitz setup
.venv/bin/es-eval judged-wands-30.jsonl --window 30 --depth 10 --inference-id pairwise-blitz --save blitz.json
```

`RERANKER_BLITZ_K` (10) is the group size and `RERANKER_BLITZ_TOP_M` (10) how many it ranks. Its answers are cached
apart from round-robin's: a pair judged among ten documents is not the same question as a pair judged alone.

To reproduce:

```sh
docker compose up -d --build && docker compose wait setup
docker compose --profile wands run --rm wands
.venv/bin/wands-judged judged-wands-all.jsonl
docker compose exec redis redis-cli FLUSHDB
.venv/bin/es-eval judged-wands-all.jsonl --per-query
```

### Bag of decisions: a window of 30

**Almost round-robin's quality, the cheapest in Jev tokens, but a new query waits for an LLM.** The same 204
queries, window 30 and top 10, ranked with [bag of decisions](https://softwaredoug.com/blog/2026/10/08/bag-of-decisions)
(Doug Turnbull, October 2026).

**How it runs here** (`reranker/questions.py`, `JevClient.decide`):
- **Questions:** an LLM writes the query's own yes/no questions, a "yes" meaning relevant. The prompts are the
  article's ([search-experiments](https://github.com/softwaredoug/search-experiments),
  `configs/ecom_decisions/bag_of_decisions.yml`) and so is the model, GPT-5. Any OpenAI-compatible API works
  (`RERANKER_QUESTIONS_URL`, `_API_KEY`, `_MODEL`, `_REASONING_EFFORT`). GPT-5 wrote 7 to 20 a query, 12.6 on average.
- **They are kept** in Redis (`reranker:questions:<hash>`, the query and its questions as JSON), keyed by the
  query, the LLM and its prompt, so each query pays for the LLM once.
- **Scoring:** one Jev request per hit carries all the questions as `noul` (yes/no) questions, with the hit's text
  as the state. The score is the mean P(yes). The article adds 10× the sum to BM25; the mean orders the same and
  stays in 0..1, so the usual BM25 tie-break applies. Its scores rarely tie, so the 0.001 tie-break can only swap hits
  closer than that.
- **Not cached:** the per-hit answers. A repeated query costs the same Jev requests again.

**Agreement with the human labels** (in-process; the GPT-5 default run through Elasticsearch gave the same 0.908):

| Measure (top 10) | BM25 | Round-robin | BlitzRank | Decisions, GPT-5 | Decisions, GPT-5 minimal reasoning |
| --- | --- | --- | --- | --- | --- |
| nDCG@10 | 0.808 | 0.921 | 0.919 | **0.908** | 0.900 |
| tau-b@10 | 0.173 | 0.323 | 0.322 | 0.305 | 0.287 |
| Pairwise accuracy@10 | 0.651 | 0.778 | 0.772 | 0.764 | 0.752 |

Against BM25, decisions with GPT-5 gains +0.101 nDCG@10 (+0.076 to +0.126; 121 queries better, 52 the same, 31
worse). That is about 0.013 below round-robin. The per-query round-robin scores were not saved, so this gap has no
paired interval. Minimal reasoning writes fewer questions (9.7 a query on average, against 12.6) and loses another 0.008.

**Delay a new query waits:**

| | p50 | p90 | max |
| --- | --- | --- | --- |
| Decisions, GPT-5: questions + Jev (in-process) | 17.5 s | 25.1 s | 34.9 s |
| Decisions, GPT-5 minimal reasoning: questions + Jev (in-process) | 2.5 s | 3.1 s | 18.8 s |
| **Decisions, questions already kept: added through Elasticsearch** | **+410 ms** | +467 ms | +759 ms |
| Round-robin, added through Elasticsearch | +4,898 ms | +5,029 ms | +5,206 ms |
| BlitzRank, added through Elasticsearch | +1,133 ms | +1,296 ms | +1,697 ms |

Once a query's questions are kept, scoring is 30 Jev requests at once (2 rounds at 16 in flight), and the delay is
the lowest of the three rankers. A repeated query waits the same +408 ms, since the per-hit answers are not cached,
while round-robin and BlitzRank drop to about +53 ms. A query never asked before waits for the LLM first. GPT-5 with
its default reasoning wrote about 2,200 output tokens per query and took 17 s, which is too long for live search.

**Cost per new query** (204 queries, in-process):

| | Jev requests | Jev input tokens | LLM tokens (in / out) |
| --- | --- | --- | --- |
| Round-robin | 435 | about 272,000 | – |
| BlitzRank | 5 | about 33,000 | – |
| **Decisions, GPT-5** | 30 | **about 21,700** | 102 / 2,180 |
| Decisions, GPT-5 minimal reasoning | 30 | about 18,900 | 102 / 220 |

A repeated query costs the 30 Jev requests again but no LLM call. At GPT-5's list price ($1.25 in, $10 out per
million tokens), the questions cost about 2.2 cents per query, or 0.2 cents with minimal reasoning.

**What it is good for:** live search for queries whose questions are already kept. Write them in advance for
frequent queries, or write them on the first search with a faster LLM. A cache of per-hit answers would bring a
repeated query down to round-robin's +53 ms.

To try it, set `RERANKER_QUESTIONS_API_KEY` and register an endpoint on `/v1/rerank/decisions`. Pass the window
settings to every compose command, or `run setup` recreates the reranker with the defaults (window 20):

```sh
export RERANKER_MAX_DOCUMENTS=30 RERANKER_TIME_BUDGET_SECONDS=60
docker compose up -d --build reranker
docker compose run --rm --no-deps -e RERANKER_URL=http://reranker:8000/v1/rerank/decisions \
  -e INFERENCE_ID=pairwise-decisions setup
.venv/bin/rerank-eval judged-wands-30.jsonl --ranker decisions --depth 10      # quality, tokens, cold time
.venv/bin/es-eval judged-wands-30.jsonl --window 30 --depth 10 --inference-id pairwise-decisions
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

- **Batching round-robin.** BlitzRank sends a group's pairs in one request. Round-robin still sends one pair per
  request, as the experiments measured.
- **Comparison sort** as another cheaper ranker; see [What to try next](#what-to-try-next).
- **Checked on one Elasticsearch version only**: 9.5.5, the compose stack. The endpoint is accepted, the reranked
  order is applied, and the `linear` retriever breaks ties and the fallback in BM25 order. Elasticsearch shifts the
  returned scores (+1, and negative ones into (0, 1)) but keeps their order. 8.19 has not been tried.

## Development

```sh
.venv/bin/pytest          # tests, with Jev answered by a fake (no network)
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```
