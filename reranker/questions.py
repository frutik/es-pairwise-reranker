"""Bag of decisions (Doug Turnbull, softwaredoug.com/blog/2026/10/08/bag-of-decisions): an LLM writes a query's own
yes/no questions, whose "yes" means a document is relevant. Any OpenAI-compatible chat-completions API.
The prompts are the article's (search-experiments, configs/ecom_decisions/bag_of_decisions.yml)."""

import hashlib
import json
from dataclasses import dataclass

import httpx

from reranker.config import Settings
from reranker.jev import JevError

# The article asks for a structured list (a "decisions" field); JSON mode is the portable way to get one.
_ANSWER_FORMAT = '\n\nAnswer with a JSON object: {"questions": ["...", "..."]}'


class QuestionError(JevError):
    """The question LLM could not be asked or answered something unreadable. A JevError, so search fails open."""


@dataclass
class Questions:
    questions: list[str]
    input_tokens: int = 0
    output_tokens: int = 0


def questions_rubric(settings: Settings) -> str:
    """Questions from another LLM or prompt are other questions, so they are stored apart."""
    s = settings
    parts = [
        s.questions_url,
        s.questions_model,
        s.questions_reasoning_effort,
        s.questions_system_prompt,
        s.questions_prompt,
    ]
    return hashlib.sha256("\u0000".join(parts).encode()).hexdigest()


class QuestionGenerator:
    def __init__(self, client: httpx.AsyncClient, settings: Settings):
        self._client = client
        self._settings = settings

    async def generate(self, query: str) -> Questions:
        s = self._settings
        request = {
            "model": s.questions_model,
            "messages": [
                {"role": "system", "content": s.questions_system_prompt},
                {"role": "user", "content": s.questions_prompt.format(query=query) + _ANSWER_FORMAT},
            ],
            "response_format": {"type": "json_object"},
        }
        if s.questions_reasoning_effort:
            request["reasoning_effort"] = s.questions_reasoning_effort
        try:
            response = await self._client.post(
                s.questions_url,
                json=request,
                headers={"Authorization": f"Bearer {s.questions_api_key}"} if s.questions_api_key else {},
                timeout=s.questions_timeout_seconds,
            )
        except httpx.HTTPError as error:
            raise QuestionError(f"could not reach the question LLM: {error!r}") from error
        if response.status_code != 200:
            raise QuestionError(f"question LLM answered {response.status_code}: {response.text[:300]}")
        try:
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            listed = json.loads(content)["questions"]
        except (ValueError, KeyError, IndexError, TypeError) as error:
            raise QuestionError(f"question LLM gave no readable questions: {response.text[:300]}") from error
        questions = list(dict.fromkeys(q.strip() for q in listed if isinstance(q, str) and q.strip()))
        if not questions:
            raise QuestionError(f"question LLM gave no questions for {query!r}")
        usage = body.get("usage") or {}
        return Questions(questions, int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0))
