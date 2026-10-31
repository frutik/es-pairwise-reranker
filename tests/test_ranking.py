from reranker.ranking import RoundRobin, Winner, pairs, swapped


def test_every_pair_once_lower_first():
    assert pairs(4) == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    assert len(pairs(10)) == 45
    assert pairs(1) == []


def test_round_robin_orders_by_wins_with_a_tie_as_half_and_keeps_the_search_order_between_equals():
    answers = {(0, 1): Winner.SECOND, (0, 2): Winner.TIE, (1, 2): Winner.FIRST}
    ranked = RoundRobin.from_answers(3, answers)

    assert ranked.scores == [0.5, 2.0, 0.5]
    assert ranked.order == [1, 0, 2]
    assert ranked.cycles == 0


def test_round_robin_counts_a_circle_of_wins_as_a_cycle():
    answers = {(0, 1): Winner.FIRST, (1, 2): Winner.FIRST, (0, 2): Winner.SECOND}

    assert RoundRobin.from_answers(3, answers).cycles == 1


def test_the_swap_is_the_same_for_the_same_question_and_shows_about_half_the_pairs_the_other_way_round():
    assert swapped("q", "a", "b", "r") == swapped("q", "a", "b", "r")
    flips = [swapped("query", f"doc {i}", f"doc {i + 1}", "rubric") for i in range(400)]

    assert 150 < sum(flips) < 250
