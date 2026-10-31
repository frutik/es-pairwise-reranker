import json
from pathlib import Path

from eval.es_eval import TIEBREAK_WEIGHT, percentile, reranked_search


def test_percentile_is_nearest_rank():
    values = [float(n) for n in range(1, 101)]
    assert percentile(values, 0.5) == 50
    assert percentile(values, 0.99) == 99
    assert percentile([3.0], 0.95) == 3.0


def test_the_reranked_search_has_the_shape_of_the_documented_example():
    body = reranked_search("angle bracket", 10, "pairwise-jev", "title")
    example = json.loads(Path("elasticsearch/search_example.json").read_text())

    linear, documented = body["retriever"]["linear"], example["retriever"]["linear"]
    assert [r["weight"] for r in linear["retrievers"]] == [r["weight"] for r in documented["retrievers"]]
    assert linear["retrievers"][1]["weight"] == TIEBREAK_WEIGHT
    assert linear["retrievers"][1]["normalizer"] == documented["retrievers"][1]["normalizer"]
    reranker = linear["retrievers"][0]["retriever"]["text_similarity_reranker"]
    assert set(reranker) == set(documented["retrievers"][0]["retriever"]["text_similarity_reranker"])
