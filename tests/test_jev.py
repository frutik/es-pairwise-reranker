import httpx
import pytest

from reranker.jev import JevClient, JevError
from reranker.ranking import Winner


def client(handler=None, **overrides) -> JevClient:
    options = {
        "url": "https://jev.test/v1/systemone",
        "api_key": "jev-key",
        "model": "jev-latest",
        "instructions": "Weigh relevance.",
        "tie_band": None,
        "max_document_chars": 4000,
    } | overrides
    transport = httpx.MockTransport(handler or (lambda request: httpx.Response(500)))
    return JevClient(httpx.AsyncClient(transport=transport), **options)


def answer(choice, probability=0.9):
    return {
        "answers": {"preference": {"type": "choice", "choice": choice, "probabilities": {choice: probability}}},
        "usage": {"input_tokens": 1200, "output_tokens": 3},
    }


def test_asks_one_choice_question_about_two_documents_as_quepid_does():
    body = client().request_body("angle bracket", "first doc", "second doc", swapped=False)

    assert body["model"] == "jev-latest"
    assert body["state"] == {"query": "angle bracket", "documents": {"A": "first doc", "B": "second doc"}}
    question = body["questions"]["preference"]
    assert question["type"] == "choice"
    assert question["instructions"] == "Weigh relevance.\n\nWhich document is more relevant to the query?"
    assert question["criteria"] == {
        "A": "Document A is more relevant to the query",
        "B": "Document B is more relevant to the query",
    }


def test_a_swapped_pair_shows_the_second_document_as_a_and_the_answer_is_read_back_the_right_way_round():
    jev = client()

    assert jev.request_body("q", "first", "second", swapped=True)["state"]["documents"] == {"A": "second", "B": "first"}
    assert jev.read(answer("A"), swapped=True).winner is Winner.SECOND
    assert jev.read(answer("A"), swapped=False).winner is Winner.FIRST
    assert jev.read(answer("B"), swapped=True).winner is Winner.FIRST


def test_a_near_even_answer_is_a_tie_only_when_a_tie_band_is_set():
    assert client(tie_band=0.6).read(answer("A", 0.55), swapped=False).winner is Winner.TIE
    assert client(tie_band=0.6).read(answer("A", 0.75), swapped=False).winner is Winner.FIRST
    assert client().read(answer("A", 0.51), swapped=False).winner is Winner.FIRST


def test_reads_the_tokens_used_and_refuses_an_answer_it_cannot_read():
    comparison = client().read(answer("B"), swapped=False)

    assert (comparison.input_tokens, comparison.output_tokens, comparison.probability) == (1200, 3, 0.9)
    with pytest.raises(JevError, match="no readable choice"):
        client().read({"answers": {"preference": {"choice": "C"}}}, swapped=False)


def test_cuts_long_documents_to_the_limit():
    body = client(max_document_chars=5).request_body("q", "0123456789", "short", swapped=False)

    assert body["state"]["documents"] == {"A": "01234 ...[truncated]", "B": "short"}


async def test_sends_the_key_and_says_what_went_wrong_when_jev_refuses():
    seen = {}

    def refuse(request):
        seen["authorization"] = request.headers["authorization"]
        return httpx.Response(429, text="slow down")

    with pytest.raises(JevError, match="Jev answered 429: slow down"):
        await client(refuse).compare("q", "a", "b", swapped=False)
    assert seen["authorization"] == "Bearer jev-key"


async def test_a_reply_that_is_not_json_is_a_jev_error():
    with pytest.raises(JevError, match="did not answer with JSON"):
        await client(lambda request: httpx.Response(200, text="<html>")).compare("q", "a", "b", swapped=False)
