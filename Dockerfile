FROM python:3.13-slim

WORKDIR /app
COPY pyproject.toml ./
COPY reranker ./reranker
COPY eval ./eval
RUN pip install --no-cache-dir .

EXPOSE 8000
# One worker per container: the comparison cache is per process.
CMD ["uvicorn", "reranker.app:app", "--host", "0.0.0.0", "--port", "8000"]
