import random

import httpx
import pytest
from fastapi.testclient import TestClient

from reranker.app import create_app
from reranker.blitz import TournamentGraph, blitz_sort
from reranker.config import Settings
from reranker.ranking import Winner
from tests.conftest import FakeJev, grade


def test_reach_and_ties_in_a_cycle():
    graph = TournamentGraph(4)
    for winner, loser in [(0, 1), (1, 2), (2, 1), (2, 3)]:  # 1 and 2 beat each other: tied
        graph.add(winner, loser)
    nodes = {info.node: info for info in graph.nodes()}

    assert nodes[0].in_reach == 0 and nodes[0].out_reach == 3 and nodes[0].known == 3
    assert nodes[1].scc == nodes[2].scc == frozenset({1, 2})
    assert (nodes[1].in_reach, nodes[1].out_reach, nodes[1].known) == (1, 1, 3)
    assert nodes[3].in_reach == 3


def perfect(strengths):
    async def ask(group, missing):
        return {(i, j): Winner.FIRST if strengths[i] > strengths[j] else Winner.SECOND for i, j in missing}

    return ask


async def test_finds_the_top_with_far_fewer_questions_than_every_pair():
    strengths = list(range(30))
    random.Random(1).shuffle(strengths)

    result = await blitz_sort(30, k=10, top_m=10, ask=perfect(strengths), known={})

    assert result.certain
    assert [node for node, _ in result.top] == sorted(range(30), key=lambda n: -strengths[n])[:10]
    assert [better for _, better in result.top] == list(range(10))
    assert result.asked < 30 * 29 // 2
    assert len(result.rest) == 20


async def test_answers_already_known_are_not_asked_again():
    strengths = list(range(12))
    first = await blitz_sort(12, k=6, top_m=3, ask=perfect(strengths), known={})
    everything = {
        (i, j): Winner.FIRST if strengths[i] > strengths[j] else Winner.SECOND
        for i in range(12)
        for j in range(i + 1, 12)
    }

    again = await blitz_sort(12, k=6, top_m=3, ask=perfect(strengths), known=everything)

    assert first.asked > 0 and again.asked == 0 and again.rounds == 0
    assert again.top == first.top


DOCUMENTS = [f"doc {n} #{g}" for n, g in enumerate([3, 9, 1, 7, 0, 8, 2, 6, 4, 5, 11, 10])]


async def test_the_service_ranks_the_top_with_group_requests_and_caches_them(make_reranker, fake_jev):
    settings = Settings(_env_file=None, jev_api_key="k", blitz_k=6, blitz_top_m=4)
    reranker = make_reranker(fake_jev, settings)

    outcome = await reranker.rerank("q", DOCUMENTS, ranker="blitz")

    best = sorted(range(len(DOCUMENTS)), key=lambda i: -grade(DOCUMENTS[i]))
    assert [ranked.index for ranked in outcome.results[:4]] == best[:4]
    assert outcome.results[3].score > outcome.results[4].score == outcome.results[-1].score
    assert outcome.ranker == "blitz" and outcome.rounds == len(fake_jev.requests) >= 1
    assert all(len(request["body"]["questions"]) > 1 for request in fake_jev.requests)

    again = await reranker.rerank("q", DOCUMENTS, ranker="blitz")
    assert again.asked == 0 and [r.index for r in again.results[:4]] == best[:4]


async def test_an_unknown_ranker_is_refused(make_reranker, fake_jev):
    with pytest.raises(ValueError):
        await make_reranker(fake_jev).rerank("q", DOCUMENTS, ranker="bubble")


def test_the_path_names_the_ranker():
    settings = Settings(_env_file=None, jev_api_key="k", blitz_k=6, blitz_top_m=4)
    with TestClient(create_app(settings, transport=httpx.MockTransport(FakeJev()))) as client:
        response = client.post("/v1/rerank/blitz", json={"query": "q", "documents": DOCUMENTS})
        assert response.status_code == 200
        assert response.json()["meta"]["ranker"] == "blitz"
        assert client.post("/v1/rerank/bubble", json={"query": "q", "documents": DOCUMENTS}).status_code == 404
        assert (
            client.post("/v1/rerank/round-robin", json={"query": "q", "documents": DOCUMENTS[:3]}).json()["meta"][
                "ranker"
            ]
            == "round-robin"
        )
