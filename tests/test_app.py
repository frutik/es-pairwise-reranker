import httpx
import pytest
from fastapi.testclient import TestClient

from reranker.app import create_app
from reranker.config import Settings
from tests.conftest import FakeJev


@pytest.fixture
def client():
    fake = FakeJev()
    settings = Settings(_env_file=None, jev_api_key="jev-key", service_api_key="service-key")
    with TestClient(create_app(settings, transport=httpx.MockTransport(fake))) as test_client:
        test_client.fake = fake
        yield test_client


AUTH = {"Authorization": "Bearer service-key"}


def test_answers_in_the_shape_the_elasticsearch_endpoint_reads(client):
    response = client.post("/v1/rerank", headers=AUTH, json={"query": "q", "documents": ["a #1", "b #3", "c #2"]})

    assert response.status_code == 200
    body = response.json()
    assert [result["index"] for result in body["results"]] == [1, 2, 0]
    assert [round(result["relevance_score"], 3) for result in body["results"]] == [1.0, 0.5, 0.0]
    assert body["meta"] == {"reranked": True, "comparisons": 3, "asked": 3, "cycles": 0, "reason": None}
    assert all(result["document"] is None for result in body["results"])


def test_returns_top_n_and_the_documents_when_asked(client):
    response = client.post(
        "/v1/rerank",
        headers=AUTH,
        json={"query": "q", "documents": ["a #1", "b #3", "c #2"], "top_n": 1, "return_documents": True},
    )

    assert response.json()["results"] == [{"index": 1, "relevance_score": 1.0, "document": "b #3"}]


def test_needs_the_service_key_when_one_is_set(client):
    assert client.post("/v1/rerank", json={"query": "q", "documents": ["a #1"]}).status_code == 401
    wrong = {"Authorization": "Bearer nope"}
    assert client.post("/v1/rerank", headers=wrong, json={"query": "q", "documents": ["a #1"]}).status_code == 401
    assert client.fake.requests == []


def test_refuses_a_request_without_the_fields_it_needs(client):
    assert client.post("/v1/rerank", headers=AUTH, json={"query": "q"}).status_code == 422
    assert client.post("/v1/rerank", headers=AUTH, json={"query": "q", "documents": [], "top_n": 0}).status_code == 422


def test_says_it_is_up(client):
    assert client.get("/health").json() == {"status": "ok", "model": "jev-latest", "max_documents": 20}
