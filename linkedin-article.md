# Asking an LLM "which of these two is better?" made our search results better. Here's what it cost.

Most LLM rerankers grade one document at a time: "how relevant is this to the query, 0 to 3?" That's a hard question, for models and for people. "Which of these two is more relevant?" is much easier. I wanted to know whether an order built from enough of those easy answers beats the search engine's own order, and what it costs in latency and tokens.

So I built a small reranking service for Elasticsearch and measured it.

## The setup

The service sits behind Elasticsearch's `custom` inference endpoint and is called from a `text_similarity_reranker` retriever, so reranking happens inside an ordinary search request. Elasticsearch sends the query and the top hits. For every pair of hits, the service asks a judge which of the two is more relevant. Each hit scores the share of comparisons it won.

The judge is TypeSafe Jev, a typed evaluation model. It gets a state (the query and two documents) and a typed question (A or B?), and answers with a choice and a probability. It writes no prose.

Three design rules came from earlier experiments:

- **Position bias is real.** On long legal texts, the judge preferred whichever document it saw first 78% of the time; on short product records, about 55%. So half the pairs are shown the other way round, chosen by a hash so the answer can be cached.
- **Judges contradict themselves.** A beats B, B beats C, C beats A: that happened 82 times in 991 comparisons. Counting wins over every pair (round-robin) absorbs it.
- **Fail open.** If the judge errors or runs out of time, the service returns equal scores and the search keeps its own order. A slow judge must never break search.

## Does it beat BM25?

I tested on WANDS, Wayfair's public furniture search dataset, with human labels (Exact, Partial, Irrelevant). I used every query whose BM25 top 10 is fully labelled and not all one grade: 222 of 480.

Reranking the top 10:

- nDCG@10: 0.903 → 0.951 (+0.048, 95% interval +0.032 to +0.065)
- Kendall's tau-b: 0.168 → 0.396
- Pairwise accuracy: 0.648 → 0.822
- Better on 130 queries, worse on 45

A first run on 30 random queries pointed the same way, but only one measure cleared the noise. Thirty queries weren't enough. When I looked at that run's worst losses, they came mostly from doubtful labels: for "desk for kids", "kids desk" was labelled Exact and every other "<brand> kids desk" Irrelevant.

## What it costs

Ten hits means 45 comparisons, sent 16 at a time.

- Plain BM25 search: 5 ms at p50
- Added by reranking, for a query nobody has asked before: +612 ms at p50, +817 ms at p99
- Added for a repeated query, all answers from cache: +53 ms
- About 28,000 input tokens per new query

## What the cache stores

The cache doesn't store result lists or scores. One entry is **one comparison**: for a query and two documents A and B, which one the judge preferred.

- **The key** is a SHA-256 hash of three things: the query text; the two document texts, in a fixed order so that "A vs B" and "B vs A" are the same entry; and the judge's settings (endpoint, model, instructions, tie threshold, truncation length). Change any setting and it's a different question, so everything is asked again.
- **The value** is just the winner: first, second or tie. The judge's probability and confidence aren't kept.
- **Not in the key:** the hits' positions, their BM25 scores, document ids, the other hits in the list, the time. Texts are hashed exactly as Elasticsearch sends them, with no lowercasing or trimming.
- **No content is stored.** Only hashes go into Redis, never queries or document text.

Keying on pairs rather than result lists is what makes changing results cheap. With 10 hits (45 pairs):

- Same query, same hits: nothing asked.
- Same hits in a different BM25 order: nothing asked, because the answers don't depend on position.
- One new hit in the top 10: 9 comparisons, its pairs with the other nine.
- One document's text edited: 9 comparisons, since the edited text is new text.
- The query typed differently ("Salon chair" vs "salon chair"): all 45.
- A different query with the same hits: all 45, because the judgement depends on the query.

The cache can never serve an answer for text it wasn't asked about. Entries for texts that disappear are simply never read again, and Redis evicts them, least recently used first. Every worker and replica shares the cache. If Redis is down or slow, its answers count as misses: the comparisons are asked again and search keeps working.

## Things Elasticsearch didn't tell me

Wiring this into Elasticsearch 9.5 taught me three things that aren't obvious from the docs:

1. It sends the reranker the top hits in **the order the documents were indexed, not by score**, and without their scores. "Indexed" here means Lucene's internal document order inside a shard, which has nothing to do with relevance. I added five documents one after another to a one-shard index, and they reached the reranker in exactly that order. With a window of 3, the three best BM25 hits were sent, but in the order they were added. (I didn't test several shards, or what happens after segment merges, but in neither case would the order be by score.)
2. When rerank scores are equal, it orders hits the same way, by that internal document order, not by BM25.
3. If the rerank endpoint returns an error, the **whole search fails**.

Together, these meant every tie, and every fallback, came back in a meaningless order. The fix sits in the query, not the service: wrap the reranker in a `linear` retriever that adds the original BM25 query again with a weight of 0.001. That weight is too small to override a real rerank score, but it breaks ties and restores BM25 order on fallback. The service, for its part, never puts input position into a score: equal scores mean "no opinion".

## Disaster: more candidates, better results, eight times slower

Reranking BM25's top 30 and scoring the top 10 lets good hits come up from below position 10:

- nDCG@10: 0.808 → 0.921 (+0.113, interval +0.089 to +0.138), more than twice the gain of reranking 10

But 30 hits are 435 comparisons per query. At 16 in parallel, that's about 27 rounds of judge calls:

- +4.9 s for a new query
- about 272,000 input tokens per new query

Fine for offline reranking or warming a cache. Not for live search.

## BlitzRank: the same quality at a quarter of the delay

BlitzRank (Agrawal, Nguyen and Kiela, ICML 2026) doesn't ask every pair. Each round it judges a group of documents and adds the answers to one preference graph, and transitivity fills in pairs nobody asked (A beats B, B beats C, so A beats C). Documents that beat each other in a circle count as tied. It stops as soon as the top 10 are certain, and doesn't spend anything ordering positions 11 to 30.

The original asks an LLM to order a group of documents. My judge can't order documents, but it can answer many questions in one request with the documents sent once. So each BlitzRank round became one request asking every pair in a group of 10: 45 answers in about 0.6 s.

On the same 204 queries, top 30, scored on the top 10:

- nDCG@10: round-robin 0.921, BlitzRank 0.919 (difference −0.002, interval −0.008 to +0.004)
- Added delay for a new query: +4.9 s → +1.13 s at p50
- Pairs asked: 435 → 195, in 5 sequential rounds
- Input tokens per new query: about 272,000 → about 33,000

Same quality, 4× less delay, 8× fewer tokens. The paper reports 7× fewer tokens than pairwise ranking with other models, so it carried over to a different kind of judge.

## What I'd keep in mind

- This is one dataset and one simple BM25 baseline. A stronger first-stage search leaves less to gain.
- Labels are noisy. Some "losses" are the labels being wrong.
- 1.1 s for a new query is still a second. Repeated queries are cheap; new ones aren't.
- If your judge's model name is an alias like `-latest`, pin a version. Otherwise the cache will quietly mix answers from two models.
- Measure on your own judged queries before sending traffic through it. The tooling to do that is in the repo.

The code, the Docker Compose stack (Elasticsearch, the reranker, Redis and a fake judge for testing without a key), the WANDS loader and the evaluation tools are on GitHub: https://github.com/frutik/es-pairwise-reranker
