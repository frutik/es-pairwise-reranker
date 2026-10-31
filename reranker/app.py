"""The HTTP service Elasticsearch's `custom` inference endpoint calls (see elasticsearch/ and the README).

POST /v1/rerank  {"query": "...", "documents": ["...", ...], "top_n": 10}
              -> {"results": [{"index": 3, "relevance_score": 0.89}, ...], "meta": {...}}

Results are best first. relevance_score is the share of the other documents a document beat (a tie is half), so it is
for sorting one query's hits, not for comparing across queries.
"""

import hmac
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from reranker.cache import open_cache
from reranker.config import Settings
from reranker.jev import JevClient
from reranker.service import Reranker, rubric


class RerankRequest(BaseModel):
    query: str
    documents: list[str]
    top_n: int | None = Field(default=None, ge=1)
    return_documents: bool = False


class RerankResult(BaseModel):
    index: int
    relevance_score: float
    document: str | None = None


class RerankResponse(BaseModel):
    results: list[RerankResult]
    meta: dict


def create_app(settings: Settings | None = None, transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    """transport: for tests, to answer Jev's requests without the network."""
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with (
            httpx.AsyncClient(timeout=settings.request_timeout_seconds, transport=transport) as client,
            open_cache(settings, rubric(settings)) as cache,
        ):
            jev = JevClient(
                client,
                url=settings.jev_url,
                api_key=settings.jev_api_key,
                model=settings.jev_model,
                instructions=settings.instructions,
                tie_band=settings.tie_band,
                max_document_chars=settings.max_document_chars,
            )
            app.state.reranker = Reranker(jev, cache, settings)
            yield

    app = FastAPI(title="Pairwise reranker", lifespan=lifespan)

    def authorised(authorization: str | None = Header(default=None)) -> None:
        if not settings.service_api_key:
            return
        expected = f"Bearer {settings.service_api_key}"
        if not authorization or not hmac.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="A valid Authorization: Bearer key is needed.")

    @app.get("/health")
    async def health() -> dict:
        cache = "redis" if settings.redis_url else "memory"
        return {"status": "ok", "model": settings.jev_model, "max_documents": settings.max_documents, "cache": cache}

    @app.post("/v1/rerank", response_model=RerankResponse, dependencies=[Depends(authorised)])
    async def rerank(request: RerankRequest) -> RerankResponse:
        outcome = await app.state.reranker.rerank(request.query, request.documents, request.top_n)
        results = [
            RerankResult(
                index=ranked.index,
                relevance_score=ranked.score,
                document=request.documents[ranked.index] if request.return_documents else None,
            )
            for ranked in outcome.results
        ]
        meta = {
            "reranked": outcome.reranked,
            "comparisons": outcome.comparisons,
            "asked": outcome.asked,
            "cycles": outcome.cycles,
            "reason": outcome.reason,
        }
        return RerankResponse(results=results, meta=meta)

    return app


app = create_app()
