"""Turning pairwise answers into an order: round-robin, as in Quepid's PairRanking::RoundRobin, the method that beat
pointwise grading on the Haystack Icecat book (see README, "Evidence")."""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from itertools import combinations


class Winner(Enum):
    FIRST = "first"  # the document listed first in the pair (the lower index)
    SECOND = "second"
    TIE = "tie"


def pairs(count: int) -> list[tuple[int, int]]:
    """Every pair of positions once, the lower first: n(n-1)/2 of them."""
    return list(combinations(range(count), 2))


def swapped(query: str, first_text: str, second_text: str, rubric: str) -> bool:
    """Whether to show the second document first. Judges lean towards whichever they see first, so half the pairs are
    shown the other way round. It is a function of what is compared rather than a coin, so an answer can be cached
    and repeated."""
    digest = hashlib.sha256("\u0000".join([rubric, query, first_text, second_text]).encode()).hexdigest()
    return int(digest, 16) % 2 == 1


@dataclass
class RoundRobin:
    """Scores and order from every pair's winner: a win is 1, a tie is half each. Equal scores keep the search's
    order. cycles counts three documents that beat each other in a circle, a sign of an unsure judge."""

    scores: list[float]
    order: list[int]
    cycles: int

    @classmethod
    def from_answers(cls, count: int, answers: Mapping[tuple[int, int], Winner]) -> "RoundRobin":
        wins = [0.0] * count
        beats = [[False] * count for _ in range(count)]
        for (i, j), winner in answers.items():
            if winner is Winner.FIRST:
                wins[i] += 1
                beats[i][j] = True
            elif winner is Winner.SECOND:
                wins[j] += 1
                beats[j][i] = True
            else:
                wins[i] += 0.5
                wins[j] += 0.5
        order = sorted(range(count), key=lambda index: (-wins[index], index))
        return cls(scores=wins, order=order, cycles=_cycles(beats))


def _cycles(beats: list[list[bool]]) -> int:
    count = 0
    for i, j, k in combinations(range(len(beats)), 3):
        if (beats[i][j] and beats[j][k] and beats[k][i]) or (beats[j][i] and beats[k][j] and beats[i][k]):
            count += 1
    return count
