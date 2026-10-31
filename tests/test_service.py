import asyncio

import httpx

from reranker.config import Settings
from reranker.service import rubric
from tests.conftest import FakeJev, grade

DOCUMENTS = ["chair #1", "angle bracket #3", "shelf bracket #2", "lamp #0"]


async def test_orders_the_hits_by_how_many_others_each_beat_best_first(make_reranker, fake_jev):
    outcome = await make_reranker(fake_jev).rerank("angle bracket", DOCUMENTS)

    assert outcome.reranked
    assert [ranked.index for ranked in outcome.results] == [1, 2, 0, 3]
    assert [round(ranked.score, 3) for ranked in outcome.results] == [1.0, 0.667, 0.333, 0.0]
    assert (outcome.comparisons, outcome.asked, outcome.cycles) == (6, 6, 0)
    assert len(fake_jev.requests) == 6


async def test_shows_about_half_the_pairs_the_other_way_round_and_still_reads_them_right(make_reranker, fake_jev):
    documents = [f"doc {i} #{i}" for i in range(12)]
    outcome = await make_reranker(fake_jev).rerank("q", documents)

    # A pair is (lower index, higher index) and here the grade is the index, so a swapped pair shows the higher first.
    shown = [request["body"]["state"]["documents"] for request in fake_jev.requests]
    swapped_pairs = sum(grade(documents["A"]) > grade(documents["B"]) for documents in shown)
    assert 0.3 * len(shown) < swapped_pairs < 0.7 * len(shown)
    assert [ranked.index for ranked in outcome.results] == list(reversed(range(12)))


async def test_asks_about_a_pair_once_and_answers_the_same_query_again_from_the_cache(make_reranker, fake_jev):
    reranker = make_reranker(fake_jev)
    await reranker.rerank("angle bracket", DOCUMENTS)
    again = await reranker.rerank("angle bracket", list(reversed(DOCUMENTS)))

    assert again.asked == 0
    assert len(fake_jev.requests) == 6
    assert [DOCUMENTS[3 - ranked.index] for ranked in again.results] == [
        "angle bracket #3",
        "shelf bracket #2",
        "chair #1",
        "lamp #0",
    ]


async def test_reranks_only_the_first_max_documents_and_puts_the_rest_below_with_equal_scores(make_reranker, fake_jev):
    settings = Settings(_env_file=None, jev_api_key="k", max_documents=3)
    outcome = await make_reranker(fake_jev, settings).rerank("q", ["a #0", "b #2", "c #1", "d #9", "e #8"])

    assert [ranked.index for ranked in outcome.results] == [1, 2, 0, 3, 4]
    assert len(fake_jev.requests) == 3
    # Equal, so the caller's own order (in Elasticsearch, the original query's) decides between them.
    assert outcome.results[2].score > outcome.results[3].score == outcome.results[4].score
    assert outcome.beyond_window == 2


async def test_returns_only_top_n(make_reranker, fake_jev):
    outcome = await make_reranker(fake_jev).rerank("q", DOCUMENTS, top_n=2)

    assert [ranked.index for ranked in outcome.results] == [1, 2]


async def test_equal_wins_are_exactly_equal_scores_listed_in_input_order(make_reranker):
    def always_a_tie(request):
        return httpx.Response(200, json={"answers": {"preference": {"choice": "A", "probabilities": {"A": 0.5}}}})

    settings = Settings(_env_file=None, jev_api_key="k", tie_band=0.6)
    outcome = await make_reranker(always_a_tie, settings).rerank("q", ["x", "y", "z"])

    assert [ranked.index for ranked in outcome.results] == [0, 1, 2]
    assert outcome.results[0].score == outcome.results[1].score == outcome.results[2].score


async def test_falls_back_to_the_search_order_when_jev_fails(make_reranker):
    outcome = await make_reranker(FakeJev(status=503)).rerank("q", DOCUMENTS)

    assert not outcome.reranked
    assert [ranked.index for ranked in outcome.results] == [0, 1, 2, 3]
    # No opinion: every hit the same score, so Elasticsearch keeps the original query's order (see search_example).
    assert len({ranked.score for ranked in outcome.results}) == 1
    assert "Jev answered 503" in outcome.reason


async def test_falls_back_to_the_search_order_when_the_time_budget_runs_out(make_reranker):
    async def slow(request):
        await asyncio.sleep(1)
        return httpx.Response(200, json={})

    settings = Settings(_env_file=None, jev_api_key="k", time_budget_seconds=0.05)
    outcome = await make_reranker(slow, settings).rerank("q", DOCUMENTS)

    assert not outcome.reranked
    assert [ranked.index for ranked in outcome.results] == [0, 1, 2, 3]
    assert "time budget" in outcome.reason


async def test_no_documents_or_one_needs_no_judge(make_reranker, fake_jev):
    reranker = make_reranker(fake_jev)

    assert (await reranker.rerank("q", [])).results == []
    assert [ranked.index for ranked in (await reranker.rerank("q", ["only #1"])).results] == [0]
    assert fake_jev.requests == []


def test_another_judge_url_is_another_rubric():
    real = Settings(_env_file=None)
    assert rubric(real) != rubric(Settings(_env_file=None, jev_url="http://fake-jev:9000"))
    assert rubric(real) == rubric(Settings(_env_file=None))
