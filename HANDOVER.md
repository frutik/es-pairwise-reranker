# Handover: es-pairwise-reranker

*Written 2026-10-31, at the end of the first session. Read this, then [README.md](README.md), to pick the work up.*

## Where things stand

A working first version of an Elasticsearch rerank service that orders hits by asking TypeSafe Jev "which of these two
documents is more relevant to the query?" for every pair, then ranks by wins. The code, tests, Elasticsearch examples,
Dockerfile and README are written. **Nothing is committed yet** (the repo has no commits) and there is no remote.

| | State |
| --- | --- |
| Service (`reranker/`) | Done: `POST /v1/rerank`, `GET /health`, optional bearer key |
| Offline evaluation (`eval/`, `rerank-eval`) | Done, never run against real Jev |
| Tests | 33 passing, Jev faked (`.venv/bin/pytest`); ruff clean |
| Smoke test | Real server run with Jev unreachable: health, 401 without key, fall-back to search order all work |
| Real Jev | **Not tried** (no API key in the session) |
| Real Elasticsearch | **Not tried**: `elasticsearch/inference_endpoint.json` follows Elastic's docs for the `custom` service only |
| Latency, cost on real data | **Unknown** |

## Decisions made, and why

The owner chose these (asked explicitly):

- **Standalone repo**, not inside Quepid: Elasticsearch calls it on every reranked search, so it must not depend on
  Quepid being up. Quepid stays the place to evaluate it.
- **Jev as the judge**, because the experiments measured Jev.
- **Live search and offline evaluation from the start.**
- **Python 3.13 + FastAPI + httpx.**
- **Elasticsearch 8.19+ / 9.1+**, so the `custom` inference service and the `text_similarity_reranker` retriever exist.

Made while building, to stay with what was measured:

- **One comparison per Jev request**, the same question and judging instructions Quepid's experiments used. Jev can
  batch questions (far fewer requests), but batching was never measured, so it is left for later on purpose.
- **Round-robin over every pair**, tie = half a win, equal scores keep the search order: the method that won.
- **Half the pairs shown swapped** against position bias, chosen by a hash (not random) so answers are cacheable.
- **Fail open**: on any Jev error or when `time_budget_seconds` runs out, return the search's order. A slow judge must
  never break search. `meta.reranked` says which happened.
- **Scores** = share of the other hits beaten (0..1), minus a tiny position tiebreak. For sorting one query's hits
  only; not comparable across queries.
- **Window** capped at `max_documents` (20 = 190 comparisons); hits past it keep their order below.
- **Cache** in memory per process (LRU, 100k entries), keyed by model + instructions + tie band + truncation + query +
  the two texts in canonical order.

## Where it came from (Quepid)

The method, numbers and Jev details come from Quepid, branch `feature/pairwise-judge` in `~/work/quepid`:

| Here | Quepid source |
| --- | --- |
| `reranker/jev.py` | `app/services/llm_judge_adapters/jev.rb` (`comparison_envelope`, `apply_comparison`) |
| `reranker/ranking.py` | `app/services/pair_ranking/round_robin.rb`, `app/services/pairwise_swap.rb` |
| `reranker/cache.py` | the `PreferenceComparison` table used by `app/services/pairwise_oracle.rb` |
| `reranker/config.py` instructions | `JEV_SYSTEM_PROMPT` in `app/models/llm_provider.rb`, minus the rating-scale sentence |
| README "Evidence" | `docs/todo/pairwise_judges_findings.md`, ADR `docs/adr/0005-pairwise-and-listwise-ranking-judges.md` |
| Pass bars | `docs/todo/pairwise_g1_bar.md`, `pairwise_g1_bar_wands.md` |

**Differences from the experiments** (they could change results, so check them when evaluating):

1. **Documents are plain strings here.** Quepid sent each document as an object of its fields (title, brand, ...), so
   Jev saw field names. Elasticsearch's reranker sends one field's text. Index a combined field if titles alone are
   too thin.
2. **The swap hash uses the texts**, Quepid's used the pair ids. Same purpose, different pairs end up swapped.
3. **The instructions** drop Quepid's sentence about the rating scale (a comparison has none).
4. **No information need**: Quepid could add one to the state; Elasticsearch has no such field.

## Jev, for reference

- `POST https://api.typesafe.ai/v1/systemone`, header `Authorization: Bearer <key>` (keys: console.typesafe.ai/keys).
- Body: `{"model": "jev-latest", "state": {"query": ..., "documents": {"A": ..., "B": ...}}, "questions":
  {"preference": {"type": "choice", "instructions": ..., "criteria": {"A": "...", "B": "..."}}}}`.
- Answer: `answers.preference.choice` (`"A"`/`"B"`), `.probabilities`, `.confidence`; `usage.input_tokens`.
- Limits (from Quepid's adapter): 64k tokens a request, 32k for state + longest question; documents are cut to 4,000
  characters each (`RERANKER_MAX_DOCUMENT_CHARS`).
- In the experiments: about 670 tokens a comparison on product records, 1,200 for two long documents.

## Next steps, in order

1. **Commit** what is here (`git add -A && git commit`), and create a remote if wanted.
2. **Run the evaluation on real data with real Jev** (`RERANKER_JEV_API_KEY` in `.env`), 20–30 queries first:
   `.venv/bin/rerank-eval judged.jsonl --limit 20`. The input format is in the README. Sources of judged data:
   - your own judged queries (best: this is what decides whether to ship);
   - Quepid's Haystack rating party (`~/work/quepid/db/sample_data/haystack_rating_party_*.json.zip`) to reproduce the
     published numbers (tau-b 0.466, nDCG@10 0.922 on its 30-query sample);
   - WANDS (`~/work/quepid/tmp/datasets/wands/`, labels 0–2).
   No converter to `judged.jsonl` exists yet; a small script per source is the next piece of code.
   Note the time per query as you go: that is the live-latency estimate.
3. **Wire it into an 8.19+/9.1+ cluster**: run the service, `PUT _inference/rerank/pairwise-jev` with
   `elasticsearch/inference_endpoint.json` (set the URL and key), try `elasticsearch/search_example.json`. Check that
   the `custom` service accepts the request template and JSONPaths as written; adjust to your version if not. Keep
   `time_budget_seconds` below Elasticsearch's inference timeout.
4. **Compare in Quepid**: two search endpoints on the same index (with and without the retriever), on a rated case.
5. **Speed, if latency is too high**: batch a query's pairs into one or a few Jev requests (documents in the state once,
   one question per pair). Measure it against the one-per-request version before switching, since the experiments
   did not test it.
6. **Shared cache** (Redis) if more than one worker or replica runs.
7. **Cheaper ranking** for windows above ~20: comparison sort (about n log n comparisons) — Quepid has
   `app/services/pair_ranking/comparison_sort.rb` to port.

## Risks and open questions

- **Evidence is thin**: one clear win (Haystack Icecat), one inconclusive (WANDS). Step 2 decides whether this is worth
  shipping.
- **Latency**: 45 parallel Jev calls for 10 hits, uncached. Likely a second or more per search; unmeasured.
- **Cost**: roughly 45 × 670 ≈ 30k tokens per uncached query of 10 hits. The cache only helps repeated queries.
- **Jev rate limits** under parallel load are unknown; `RERANKER_CONCURRENCY` (16) bounds it per rerank.
- **Position bias** was large on long texts (78% for the first shown); the swap averages it out but costs accuracy.

## Running it

```sh
cd ~/work/es-pairwise-reranker
.venv/bin/pytest                                     # tests (fake Jev, no network)
cp .env.example .env                                 # then set the keys
.venv/bin/uvicorn reranker.app:app --port 8000       # the service
.venv/bin/rerank-eval judged.jsonl --limit 20        # evaluation, real Jev
```

The virtualenv was made with Homebrew's Python 3.13 (`/opt/homebrew/bin/python3.13`). The `python3` first on the PATH
belongs to another project's virtualenv, so recreate this one with the Homebrew Python if needed.

## Repo layout

```
reranker/   config.py (settings), jev.py (Jev client), ranking.py (round-robin, swap), cache.py,
            service.py (rerank + fail-open), metrics.py (nDCG, tau-b, pairwise accuracy), app.py (FastAPI)
eval/       evaluate.py (rerank-eval)
tests/      conftest.py (FakeJev: prefers the text ending in the higher "#grade"), one file per module
elasticsearch/  inference_endpoint.json, search_example.json
```
