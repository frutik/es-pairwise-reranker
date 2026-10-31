"""Comparisons already paid for, in memory. The key is the question (model, instructions, tie band), the query and the
two texts, in a fixed order, so A against B and B against A are one entry. Per process: several workers each keep
their own."""

import hashlib
from collections import OrderedDict

from reranker.ranking import Winner

_FLIPPED = {Winner.FIRST: Winner.SECOND, Winner.SECOND: Winner.FIRST, Winner.TIE: Winner.TIE}


class ComparisonCache:
    def __init__(self, size: int, rubric: str):
        self._size = size
        self._rubric = rubric
        self._entries: OrderedDict[str, Winner] = OrderedDict()

    def get(self, query: str, first: str, second: str) -> Winner | None:
        key, flipped = self._key(query, first, second)
        winner = self._entries.get(key)
        if winner is None:
            return None
        self._entries.move_to_end(key)
        return _FLIPPED[winner] if flipped else winner

    def put(self, query: str, first: str, second: str, winner: Winner) -> None:
        if self._size <= 0:
            return
        key, flipped = self._key(query, first, second)
        self._entries[key] = _FLIPPED[winner] if flipped else winner
        self._entries.move_to_end(key)
        while len(self._entries) > self._size:
            self._entries.popitem(last=False)

    def __len__(self) -> int:
        return len(self._entries)

    def _key(self, query: str, first: str, second: str) -> tuple[str, bool]:
        flipped = second < first
        low, high = (second, first) if flipped else (first, second)
        key = hashlib.sha256("\u0000".join([self._rubric, query, low, high]).encode()).hexdigest()
        return key, flipped
