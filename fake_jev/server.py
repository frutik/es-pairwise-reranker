"""A stand-in for TypeSafe Jev, for running the stack without a key or without paying for comparisons. Only Python's
standard library, so it runs in any python image.

It answers the comparison requests the reranker sends (POST, any path) in Jev's shape. The judge is word overlap:
the document sharing more distinct words with the query wins, and a draw goes to the document shown first, with
probability 0.5 (so RERANKER_TIE_BAND above 0.5 turns draws into ties). Nothing about it is a real relevance
judgement: it is for checking the wiring, the cache, timeouts and fail-open, not for measuring quality.

    FAKE_JEV_PORT=9000          port to listen on
    FAKE_JEV_DELAY_SECONDS=0    wait this long before each answer (to try the time budget)
    FAKE_JEV_FAIL=0             answer every comparison with 503 when 1 (to try fail-open)
"""

import json
import os
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

WORD = re.compile(r"\w+")


def words(text: str) -> set[str]:
    return set(WORD.findall(text.lower()))


def judge(query: str, a: str, b: str) -> tuple[str, float]:
    """The choice ("A" or "B") and its probability."""
    wanted = words(query)
    overlap_a, overlap_b = len(wanted & words(a)), len(wanted & words(b))
    if overlap_a == overlap_b:
        return "A", 0.5
    return ("A", 0.9) if overlap_a > overlap_b else ("B", 0.9)


def answer(body: dict) -> dict:
    """Answers every question of the request: one pair ("A"/"B"), or a BlitzRank group's pairs ("D1".."Dn"). Each
    question names its two documents in its criteria."""
    documents = body["state"]["documents"]
    answers = {}
    for question_id, question in body["questions"].items():
        first, second = list(question["criteria"])
        choice, probability = judge(body["state"]["query"], documents[first], documents[second])
        choice = first if choice == "A" else second
        answers[question_id] = {
            "type": "choice",
            "choice": choice,
            "probabilities": {choice: probability},
            "confidence": probability,
        }
    return {"answers": answers, "usage": {"input_tokens": 0, "output_tokens": 0}}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self._send(200, {"status": "ok", "fake": True})

    def do_POST(self) -> None:
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            result = answer(body)
        except (ValueError, KeyError, TypeError) as error:
            self._send(400, {"error": f"not a comparison request: {error}"})
            return
        time.sleep(float(os.environ.get("FAKE_JEV_DELAY_SECONDS", "0")))
        if os.environ.get("FAKE_JEV_FAIL") == "1":
            self._send(503, {"error": "failing on purpose (FAKE_JEV_FAIL=1)"})
            return
        self._send(200, result)

    def _send(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args) -> None:  # one short line a request, not the default chatter
        print(f"{self.command} {self.path} {args[1] if len(args) > 1 else ''}", flush=True)


def main() -> None:
    port = int(os.environ.get("FAKE_JEV_PORT", "9000"))
    print(f"fake Jev listening on :{port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
