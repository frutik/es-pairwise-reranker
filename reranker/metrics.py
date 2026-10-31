"""How well an order agrees with human grades: the measures the pairwise experiments used. Each takes the grades in
the order being judged, best-placed first."""

import math
from itertools import combinations


def ndcg(grades: list[float], k: int = 10) -> float | None:
    """nDCG@k with gains 2^grade - 1. None when no document has a positive grade (nothing to find)."""
    ideal = _dcg(sorted(grades, reverse=True)[:k])
    if ideal == 0:
        return None
    return _dcg(grades[:k]) / ideal


def _dcg(grades: list[float]) -> float:
    return sum((2**grade - 1) / math.log2(position + 2) for position, grade in enumerate(grades))


def pairwise_accuracy(grades: list[float]) -> float | None:
    """Of the pairs people graded differently, the share placed the right way round. None when all grades are equal."""
    right = total = 0
    for earlier, later in combinations(grades, 2):
        if earlier == later:
            continue
        total += 1
        right += earlier > later
    return right / total if total else None


def kendall_tau_b(grades: list[float]) -> float | None:
    """Kendall's tau-b between the placement (first is best) and the grades. None when the grades are all equal."""
    concordant = discordant = tied_grades = 0
    for earlier, later in combinations(grades, 2):
        if earlier == later:
            tied_grades += 1
        elif earlier > later:
            concordant += 1
        else:
            discordant += 1
    pairs = len(grades) * (len(grades) - 1) // 2
    denominator = math.sqrt(pairs * (pairs - tied_grades))
    return (concordant - discordant) / denominator if denominator else None
