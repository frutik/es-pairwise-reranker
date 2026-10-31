from reranker.cache import ComparisonCache
from reranker.ranking import Winner


def test_a_pair_asked_the_other_way_round_is_the_same_answer_read_back_flipped():
    cache = ComparisonCache(10, "rubric")
    cache.put("q", "apple", "pear", Winner.FIRST)

    assert cache.get("q", "apple", "pear") is Winner.FIRST
    assert cache.get("q", "pear", "apple") is Winner.SECOND
    assert len(cache) == 1


def test_another_query_or_question_is_not_the_same_entry():
    cache = ComparisonCache(10, "rubric")
    cache.put("q", "apple", "pear", Winner.FIRST)

    assert cache.get("other query", "apple", "pear") is None
    assert ComparisonCache(10, "another rubric").get("q", "apple", "pear") is None


def test_forgets_the_least_recently_used_beyond_its_size():
    cache = ComparisonCache(2, "rubric")
    cache.put("q", "a", "b", Winner.FIRST)
    cache.put("q", "a", "c", Winner.FIRST)
    cache.get("q", "a", "b")
    cache.put("q", "a", "d", Winner.TIE)

    assert cache.get("q", "a", "c") is None
    assert cache.get("q", "a", "b") is Winner.FIRST
    assert cache.get("q", "d", "a") is Winner.TIE


def test_a_size_of_zero_keeps_nothing():
    cache = ComparisonCache(0, "rubric")
    cache.put("q", "a", "b", Winner.FIRST)

    assert len(cache) == 0
