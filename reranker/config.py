"""Settings, read from the environment (or a .env file). Every name is prefixed RERANKER_."""

from pydantic_settings import BaseSettings, SettingsConfigDict

# The judging instructions Quepid's Jev judge used in the pairwise experiments, minus the sentence about a rating
# scale, which a comparison does not have. Changing them changes the question, so the cache keys on them.
DEFAULT_INSTRUCTIONS = (
    "Judge how well the document satisfies the user's query and, when one is given, the stated information need.\n\n"
    "Weigh what the document is actually about and how specific it is to the query -- not its length, style or "
    "formatting. A document that merely mentions the query's words without addressing it is not relevant."
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RERANKER_", env_file=".env", extra="ignore")

    # TypeSafe Jev.
    jev_api_key: str = ""
    jev_url: str = "https://api.typesafe.ai/v1/systemone"
    jev_model: str = "jev-latest"
    instructions: str = DEFAULT_INSTRUCTIONS
    # A comparison whose winner Jev gives less than this probability counts as a tie. None: never a tie.
    tie_band: float | None = None
    # Each document's text is cut to this many characters before it is sent (Jev caps the request size).
    max_document_chars: int = 4_000

    # How many of the hits are reranked: n of them take n(n-1)/2 comparisons. Hits past this keep their order, below.
    max_documents: int = 20
    # Jev requests in flight at once, for one rerank.
    concurrency: int = 16
    # How long one Jev request may take, and how long a whole rerank may. Past the budget, the search's own order is
    # returned (fail open), so a slow or failing judge never breaks search.
    request_timeout_seconds: float = 10.0
    time_budget_seconds: float = 8.0

    # Comparisons kept, so a pair is paid for once. Without a Redis URL, in this process's memory (cache_size entries);
    # with one, in Redis, shared by every worker and replica (Redis's maxmemory policy evicts, cache_ttl_seconds
    # expires, 0 never). A Redis call that takes longer than redis_timeout_seconds counts as a miss.
    cache_size: int = 100_000
    redis_url: str = ""
    cache_ttl_seconds: int = 0
    redis_timeout_seconds: float = 0.5

    # When set, callers must send "Authorization: Bearer <this>" (Elasticsearch's inference endpoint can).
    service_api_key: str = ""
