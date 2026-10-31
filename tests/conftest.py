import json

import httpx
import pytest

from reranker.cache import MemoryCache
from reranker.config import Settings
from reranker.jev import JevClient
from reranker.service import Reranker, rubric


class FakeJev:
    """Answers Jev's comparison requests the way a perfect judge would: the document whose text ends in the higher
    grade ("... #3" beats "... #1") wins, equal grades go to the one shown first. Records every request."""

    def __init__(self, status: int = 200):
        self.requests: list[dict] = []
        self.status = status

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append({"body": body, "headers": dict(request.headers)})
        if self.status != 200:
            return httpx.Response(self.status, text="overloaded")
        shown = body["state"]["documents"]
        answers = {}
        # One question (a single pair, "A"/"B") or many (a BlitzRank group, "D1".."Dn"): each names its two
        # documents in its criteria, the one listed first winning a draw.
        for question_id, question in body["questions"].items():
            first, second = list(question["criteria"])
            choice = first if grade(shown[first]) >= grade(shown[second]) else second
            answers[question_id] = {
                "type": "choice",
                "choice": choice,
                "probabilities": {choice: 0.9},
                "confidence": 0.8,
            }
        return httpx.Response(200, json={"answers": answers, "usage": {"input_tokens": 600, "output_tokens": 4}})


def grade(text: str) -> int:
    return int(text.rsplit("#", 1)[1])


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, jev_api_key="jev-key", max_documents=20, time_budget_seconds=5)


@pytest.fixture
def fake_jev() -> FakeJev:
    return FakeJev()


@pytest.fixture
async def make_reranker(settings):
    clients = []

    def make(handler, settings: Settings = settings) -> Reranker:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        clients.append(client)
        jev = JevClient(
            client,
            url=settings.jev_url,
            api_key=settings.jev_api_key,
            model=settings.jev_model,
            instructions=settings.instructions,
            tie_band=settings.tie_band,
            max_document_chars=settings.max_document_chars,
        )
        return Reranker(jev, MemoryCache(settings.cache_size, rubric(settings)), settings)

    yield make
    for client in clients:
        await client.aclose()
