from fake_jev.server import answer, judge
from reranker.jev import JevClient
from reranker.ranking import Winner


def test_the_document_sharing_more_query_words_wins_and_a_draw_is_even():
    assert judge("angle bracket", "Patio chair", "Angle bracket, steel") == ("B", 0.9)
    assert judge("angle bracket", "bracket", "angle") == ("A", 0.5)


def test_answers_what_the_reranker_sends_in_a_shape_it_reads():
    client = JevClient(None, url="", api_key="", model="m", instructions="i", tie_band=0.6, max_document_chars=100)
    body = client.request_body("angle bracket", "Patio chair", "Angle bracket", swapped=True)

    assert client.read(answer(body), swapped=True).winner is Winner.SECOND
    assert client.read(answer(client.request_body("q", "x", "y", swapped=False)), swapped=False).winner is Winner.TIE
