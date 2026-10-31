#!/bin/sh
# Creates (or re-creates) the pairwise-jev rerank inference endpoint, pointing Elasticsearch at the reranker.
# Run by docker compose's `setup` service; also usable by hand:
#   ES_URL=http://localhost:9200 RERANKER_URL=http://host.docker.internal:8000/v1/rerank \
#     RERANKER_SERVICE_API_KEY=... sh elasticsearch/setup.sh
set -eu

ES_URL="${ES_URL:-http://elasticsearch:9200}"
RERANKER_URL="${RERANKER_URL:-http://reranker:8000/v1/rerank}"
INFERENCE_ID="${INFERENCE_ID:-pairwise-jev}"
: "${RERANKER_SERVICE_API_KEY:?set RERANKER_SERVICE_API_KEY}"

echo "waiting for Elasticsearch at $ES_URL"
until curl -fsS "$ES_URL/_cluster/health?wait_for_status=yellow&timeout=5s" >/dev/null 2>&1; do sleep 2; done

# Re-running replaces the endpoint, so a changed URL or key takes effect.
curl -sS -o /dev/null -X DELETE "$ES_URL/_inference/rerank/$INFERENCE_ID?force=true" || true

body=$(cat <<EOF
{
  "service": "custom",
  "service_settings": {
    "secret_parameters": { "api_key": "$RERANKER_SERVICE_API_KEY" },
    "url": "$RERANKER_URL",
    "headers": { "Authorization": "Bearer \${api_key}", "Content-Type": "application/json" },
    "request": "{\"query\": \${query}, \"documents\": \${input}}",
    "response": {
      "json_parser": {
        "relevance_score": "\$.results[*].relevance_score",
        "reranked_index": "\$.results[*].index"
      }
    }
  }
}
EOF
)

echo "creating inference endpoint $INFERENCE_ID -> $RERANKER_URL"
status=$(curl -sS -o /tmp/response.json -w '%{http_code}' -X PUT "$ES_URL/_inference/rerank/$INFERENCE_ID" \
  -H 'Content-Type: application/json' -d "$body")
cat /tmp/response.json; echo
[ "$status" = "200" ] || { echo "failed: HTTP $status"; exit 1; }
