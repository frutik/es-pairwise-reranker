import io
import math

import pytest

from eval.evaluate import evaluate_with
from reranker.metrics import kendall_tau_b, ndcg, pairwise_accuracy


def test_ndcg_is_one_for_the_ideal_order_and_lower_otherwise():
    assert ndcg([3, 2, 1, 0]) == pytest.approx(1.0)
    worse = ndcg([0, 1, 2, 3])
    expected = (0 + 1 / math.log2(3) + 3 / 2 + 7 / math.log2(5)) / (7 + 3 / math.log2(3) + 1 / 2)
    assert worse == pytest.approx(expected)
    assert ndcg([0, 0]) is None


def test_pairwise_accuracy_counts_only_pairs_people_graded_differently():
    assert pairwise_accuracy([3, 2, 2, 0]) == 1.0
    assert pairwise_accuracy([0, 3]) == 0.0
    assert pairwise_accuracy([2, 1, 3]) == pytest.approx(1 / 3)
    assert pairwise_accuracy([1, 1]) is None


def test_kendall_tau_b_handles_ties_in_the_grades():
    assert kendall_tau_b([3, 2, 1]) == pytest.approx(1.0)
    assert kendall_tau_b([1, 2, 3]) == pytest.approx(-1.0)
    assert kendall_tau_b([2, 2, 1]) == pytest.approx(2 / math.sqrt(3 * 2))
    assert kendall_tau_b([1, 1, 1]) is None


async def test_the_evaluation_compares_the_search_order_with_the_reranked_one(make_reranker, fake_jev):
    queries = [
        {
            "query": "q1",
            "documents": [{"text": "a #0", "grade": 0}, {"text": "b #2", "grade": 2}, {"text": "c #1", "grade": 1}],
        },
        {"query": "q2", "documents": [{"text": "d #1", "grade": 1}, {"text": "e #1", "grade": 1}]},
    ]
    out = io.StringIO()

    summary = await evaluate_with(make_reranker(fake_jev), queries, out)

    assert summary["reranked"]["pairwise accuracy"] == 1.0
    assert summary["search"]["pairwise accuracy"] == pytest.approx(1 / 3)
    assert summary["reranked"]["nDCG@10"] > summary["search"]["nDCG@10"]
    assert "2 queries, 4 comparisons asked, 0 fell back to the search's order" in out.getvalue()
