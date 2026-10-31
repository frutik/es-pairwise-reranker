# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

An HTTP rerank service that Elasticsearch (8.19+ / 9.1+) calls through a `custom` inference endpoint from a
`text_similarity_reranker` retriever. It asks TypeSafe Jev "which of these two documents is more relevant?" for every
pair of the top hits and orders them by wins. `README.md` covers the method, the evidence behind it and the
Elasticsearch wiring. `HANDOVER.md` covers the decisions made so far, where the code was ported from in Quepid, and
the next steps in order.

## Commands

Use the project virtualenv (currently Homebrew Python 3.14; the project needs 3.12+). Do not build it with the
`python3` first on the PATH, which belongs to another project's virtualenv.

```sh
/opt/homebrew/bin/python3.14 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pytest                                           # all tests; Jev is faked, no network
.venv/bin/pytest tests/test_service.py::test_name -q       # a single test
.venv/bin/ruff check . && .venv/bin/ruff format --check .  # lint + format (line length 120)
.venv/bin/uvicorn reranker.app:app --port 8000             # run the service (needs .env, see .env.example)
.venv/bin/rerank-eval judged.jsonl --limit 20              # offline eval: calls real Jev and costs tokens
docker compose up -d --build                               # ES 9.5 + reranker (2 workers) + Redis; `setup` registers
                                                           # the inference endpoint via elasticsearch/setup.sh
docker compose wait setup                                  # `up -d` returns before setup has registered it
docker compose --profile wands run --rm wands              # download WANDS from GitHub + index it as `wands`
RERANKER_JEV_URL=http://fake-jev:9000 docker compose --profile fake-jev up -d   # no key, no cost: fake Jev
.venv/bin/wands-judged judged-wands-all.jsonl --es http://localhost:9201   # judged queries (add --sample 30)
.venv/bin/es-eval judged-wands-all.jsonl --es http://localhost:9201       # quality + added delay through ES
```

If the reranker container is recreated, Elasticsearch keeps calling its old IP ("Connection refused"). Restart
Elasticsearch, then run `docker compose run --rm setup`.

`.env` holds a real Jev key, which compose passes to the reranker. A search through the stack without the fake-jev
profile calls real Jev and costs tokens.

Another project's Elasticsearch (`quepid-test-es`) often holds port 9200 on this machine. Use `ES_PORT=9201 docker
compose up -d` instead of stopping it.

pytest runs with `asyncio_mode = "auto"`, so async tests need no marker.

## Architecture

Request flow: `app.py` (FastAPI, `POST /v1/rerank`, optional bearer key) → `service.Reranker.rerank` →
`ranking.pairs` → `cache.ComparisonCache` lookup → `jev.JevClient.compare` for the missing pairs (parallel, bounded by
`concurrency`) → `ranking.RoundRobin` → scores.

Invariants that span modules:

- **Fail open.** Any `JevError`, or `time_budget_seconds` running out (`asyncio.wait_for`), returns equal scores for
  every hit with `reranked=False`. The service must answer 200, because an error from the endpoint makes
  Elasticsearch fail the whole search. Search must never break because of the judge. Non-`JevError` exceptions from a
  comparison are wrapped in `JevError` so they also fail open.
- **The rubric.** `service.rubric(settings)` hashes the Jev URL, model, instructions, tie band and truncation length. It seeds both
  the cache key and the swap hash, so any change to how the question is asked has to be added to the rubric.
- **Swap against position bias.** `ranking.swapped` shows about half the pairs B-first, chosen deterministically from a
  hash (not at random) so that answers can be cached. `JevClient.read` undoes the swap. Callers always deal in
  `Winner.FIRST`/`SECOND`, which refer to the lower and higher index of the pair.
- **The cache key is order-independent.** The cache stores each pair under the canonical (lexicographic) text order
  and flips the winner when a lookup comes in the other way round. `cache.py` holds the `ComparisonCache` protocol
  (async `get_many` and `put`) with two implementations: `MemoryCache` (per-process LRU) and `RedisCache` (shared,
  chosen by `open_cache` when `RERANKER_REDIS_URL` is set). Redis errors and timeouts count as misses, never as
  failures. This is the fail-open rule again.
- **Equal scores mean "no opinion"; the caller breaks them.** Elasticsearch sends hits in index order, without
  scores, and orders equal rerank scores by index (seen on 9.5.5). So the service cannot know the search's order and
  never encodes input position in a score. Ties in wins, the fallback (all 0.0) and hits past `max_documents` (all
  -1.0) get exactly equal scores. `elasticsearch/search_example.json` wraps the reranker in a `linear` retriever
  that adds the original query with weight 0.001, and that breaks them by BM25. Keep that weight below the smallest
  rerank-score gap, 0.5/(n-1). The results list is sorted by (score, input position), which is the search's order
  only for callers that send hits in that order (`rerank-eval`).
- **Scores** are the share of the other hits beaten (0..1). They only sort hits within one query and cannot be
  compared across queries.
- **One comparison per Jev request**, in the exact shape the Quepid experiments measured. Batching is deliberately not
  done yet. Measure it against the current version before you switch.

Tests: `tests/test_cache.py` runs each cache test against both stores (Redis via `fakeredis`). `tests/conftest.py`
provides `FakeJev`, an `httpx.MockTransport` handler that acts as a perfect judge. A document whose text ends in a
higher `#<grade>` wins, and on equal grades the document shown first wins. It also records every request. It is
separate from `fake_jev/server.py`, the HTTP stand-in for the compose stack, which judges by word overlap. `make_reranker(handler, settings)` builds a `Reranker` around any handler. `create_app` takes a
`transport` so the app can be tested the same way.

All settings are `RERANKER_*` environment variables (`reranker/config.py`, `.env.example`). The default judging
instructions there are Quepid's, minus its rating-scale sentence. Changing them changes the question being asked.
