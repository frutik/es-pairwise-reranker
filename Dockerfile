FROM python:3.13-slim

WORKDIR /app
COPY pyproject.toml ./
COPY reranker ./reranker
COPY eval ./eval
RUN pip install --no-cache-dir .

EXPOSE 8000
# Workers come from WEB_CONCURRENCY (default 1). Without RERANKER_REDIS_URL each worker keeps its own cache.
CMD ["uvicorn", "reranker.app:app", "--host", "0.0.0.0", "--port", "8000"]
