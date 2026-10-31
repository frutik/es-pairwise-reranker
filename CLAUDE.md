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
docker compose --profile wands run --rm wands              # download WANDS from GitHub + index it as `wands`
```

Another project's Elasticsearch (`quepid-test-es`) often holds port 9200 on this machine. Use `ES_PORT=9201 docker
compose up -d` instead of stopping it.

pytest runs with `asyncio_mode = "auto"`, so async tests need no marker.

## Architecture

Request flow: `app.py` (FastAPI, `POST /v1/rerank`, optional bearer key) → `service.Reranker.rerank` →
`ranking.pairs` → `cache.ComparisonCache` lookup → `jev.JevClient.compare` for the missing pairs (parallel, bounded by
`concurrency`) → `ranking.RoundRobin` → scores.

Invariants that span modules:

- **Fail open.** Any `JevError`, or `time_budget_seconds` running out (`asyncio.wait_for`), returns the search's own
  order with `reranked=False`. Search must never break because of the judge. Non-`JevError` exceptions from a
  comparison are wrapped in `JevError` so they also fail open.
- **The rubric.** `service.rubric(settings)` hashes model, instructions, tie band and truncation length. It seeds both
  the cache key and the swap hash, so any change to how the question is asked has to be added to the rubric.
- **Swap against position bias.** `ranking.swapped` shows about half the pairs B-first, chosen deterministically from a
  hash (not at random) so that answers can be cached. `JevClient.read` undoes the swap. Callers always deal in
  `Winner.FIRST`/`SECOND`, which refer to the lower and higher index of the pair.
- **The cache key is order-independent.** The cache stores each pair under the canonical (lexicographic) text order
  and flips the winner when a lookup comes in the other way round. `cache.py` holds the `ComparisonCache` protocol
  (async `get_many` and `put`) with two implementations: `MemoryCache` (per-process LRU) and `RedisCache` (shared,
  chosen by `open_cache` when `RERANKER_REDIS_URL` is set). Redis errors and timeouts count as misses, never as
  failures. This is the fail-open rule again.
- **Elasticsearch sends hits in index order, not score order** (seen on 9.5.5). Code that treats the input order as
  "the search's order" (fallback, tiebreak, the `max_documents` window) is therefore wrong under Elasticsearch, though
  correct in `rerank-eval`. See the README's "Not done yet".
- **Scores** are the share of the other hits beaten, minus a `1e-6 × position` tiebreak so that ties keep the search
  order once Elasticsearch sorts. Hits past `max_documents` get negative scores below everything reranked. Scores only
  sort hits within one query. They cannot be compared across queries.
- **One comparison per Jev request**, in the exact shape the Quepid experiments measured. Batching is deliberately not
  done yet. Measure it against the current version before you switch.

Tests: `tests/test_cache.py` runs each cache test against both stores (Redis via `fakeredis`). `tests/conftest.py` provides `FakeJev`, an `httpx.MockTransport` handler that acts as a perfect judge. A
document whose text ends in a higher `#<grade>` wins, and on equal grades the document shown first wins. It also
records every request. `make_reranker(handler, settings)` builds a `Reranker` around any handler. `create_app` takes a
`transport` so the app can be tested the same way.

All settings are `RERANKER_*` environment variables (`reranker/config.py`, `.env.example`). The default judging
instructions there are Quepid's, minus its rating-scale sentence. Changing them changes the question being asked.
