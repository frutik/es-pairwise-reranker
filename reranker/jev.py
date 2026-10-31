"""TypeSafe Jev, asked which of two documents is more relevant to a query. One comparison per request, in the shape
Quepid's Jev adapter sends (LlmJudgeAdapters::Jev#comparison_envelope), which is what the experiments measured."""

from dataclasses import dataclass

import httpx

from reranker.ranking import Winner

QUESTION_ID = "preference"
SHOWN_FIRST = "A"
SHOWN_SECOND = "B"
TRUNCATION_MARKER = " ...[truncated]"


class JevError(Exception):
    """Jev could not be asked, or answered something that cannot be read."""


@dataclass
class Comparison:
    winner: Winner
    probability: float | None
    input_tokens: int | None
    output_tokens: int | None


class JevClient:
    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        url: str,
        api_key: str,
        model: str,
        instructions: str,
        tie_band: float | None,
        max_document_chars: int,
    ):
        self._client = client
        self._url = url
        self._api_key = api_key
        self._model = model
        self._instructions = instructions
        self._tie_band = tie_band
        self._max_document_chars = max_document_chars

    def request_body(self, query: str, first: str, second: str, *, swapped: bool) -> dict:
        """The request for one comparison. swapped shows the second document as "A"."""
        shown_a, shown_b = (second, first) if swapped else (first, second)
        return {
            "model": self._model,
            "state": {"query": query, "documents": {SHOWN_FIRST: self._cut(shown_a), SHOWN_SECOND: self._cut(shown_b)}},
            "questions": {
                QUESTION_ID: {
                    "type": "choice",
                    "instructions": f"{self._instructions}\n\nWhich document is more relevant to the query?",
                    "criteria": {
                        SHOWN_FIRST: "Document A is more relevant to the query",
                        SHOWN_SECOND: "Document B is more relevant to the query",
                    },
                }
            },
        }

    async def compare(self, query: str, first: str, second: str, *, swapped: bool) -> Comparison:
        """Which of first and second (in the caller's terms, the swap undone) is more relevant to the query."""
        try:
            response = await self._client.post(
                self._url,
                json=self.request_body(query, first, second, swapped=swapped),
                # No key, no header: Jev answers 401, which says more than a malformed "Bearer ".
                headers={"Authorization": f"Bearer {self._api_key}"} if self._api_key else {},
            )
        except httpx.HTTPError as error:
            raise JevError(f"could not reach Jev: {error}") from error
        if response.status_code != 200:
            raise JevError(f"Jev answered {response.status_code}: {response.text[:300]}")
        try:
            body = response.json()
        except ValueError as error:
            raise JevError(f"Jev did not answer with JSON: {response.text[:300]}") from error
        return self.read(body, swapped=swapped)

    def read(self, body: dict, *, swapped: bool) -> Comparison:
        answer = (body.get("answers") or {}).get(QUESTION_ID) or {}
        chosen = answer.get("choice")
        if chosen not in (SHOWN_FIRST, SHOWN_SECOND):
            raise JevError(f"Jev gave no readable choice: {body!r:.300}")

        probability = _number((answer.get("probabilities") or {}).get(chosen))
        if self._tie_band is not None and probability is not None and probability < self._tie_band:
            winner = Winner.TIE
        else:
            first_won = (chosen == SHOWN_FIRST) != swapped
            winner = Winner.FIRST if first_won else Winner.SECOND

        usage = body.get("usage") or {}
        return Comparison(
            winner=winner,
            probability=probability,
            input_tokens=_integer(usage.get("input_tokens")),
            output_tokens=_integer(usage.get("output_tokens")),
        )

    def _cut(self, text: str) -> str:
        if len(text) <= self._max_document_chars:
            return text
        return text[: self._max_document_chars] + TRUNCATION_MARKER


def _number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and number not in (float("inf"), float("-inf")) else None


def _integer(value) -> int | None:
    number = _number(value)
    return int(number) if number is not None else None
