import fakeredis
import pytest

from reranker.cache import MemoryCache, RedisCache, open_cache
from reranker.config import Settings
from reranker.ranking import Winner


@pytest.fixture
def redis_server() -> fakeredis.FakeServer:
    return fakeredis.FakeServer()


@pytest.fixture(params=["memory", "redis"])
def make_cache(request, redis_server):
    def make(rubric: str = "rubric", size: int = 10):
        if request.param == "memory":
            return MemoryCache(size, rubric)
        return RedisCache(fakeredis.FakeAsyncRedis(server=redis_server), rubric)

    return make


async def get(cache, query: str, first: str, second: str) -> Winner | None:
    return (await cache.get_many(query, [(first, second)]))[0]


async def test_a_pair_asked_the_other_way_round_is_the_same_answer_read_back_flipped(make_cache):
    cache = make_cache()
    await cache.put("q", "apple", "pear", Winner.FIRST)

    assert await cache.get_many("q", [("apple", "pear"), ("pear", "apple"), ("apple", "plum")]) == [
        Winner.FIRST,
        Winner.SECOND,
        None,
    ]


async def test_another_query_or_question_is_not_the_same_entry(make_cache):
    cache = make_cache()
    await cache.put("q", "apple", "pear", Winner.FIRST)

    assert await get(cache, "other query", "apple", "pear") is None
    assert await get(make_cache("another rubric"), "q", "apple", "pear") is None


async def test_no_pairs_is_no_answers(make_cache):
    assert await make_cache().get_many("q", []) == []


async def test_redis_is_shared_between_caches_on_the_same_server(redis_server):
    one = RedisCache(fakeredis.FakeAsyncRedis(server=redis_server), "rubric")
    other = RedisCache(fakeredis.FakeAsyncRedis(server=redis_server), "rubric")
    await one.put("q", "a", "b", Winner.TIE)

    assert await get(other, "q", "b", "a") is Winner.TIE


async def test_redis_entries_expire_when_a_ttl_is_set(redis_server):
    client = fakeredis.FakeAsyncRedis(server=redis_server)
    await RedisCache(client, "rubric", ttl_seconds=60).put("q", "a", "b", Winner.FIRST)

    (key,) = await client.keys(RedisCache.KEY_PREFIX + "*")
    assert 0 < await client.ttl(key) <= 60


async def test_a_redis_that_is_down_is_a_miss_not_an_error(redis_server):
    redis_server.connected = False
    cache = RedisCache(fakeredis.FakeAsyncRedis(server=redis_server), "rubric")

    await cache.put("q", "a", "b", Winner.FIRST)
    assert await cache.get_many("q", [("a", "b"), ("a", "c")]) == [None, None]


async def test_memory_forgets_the_least_recently_used_beyond_its_size():
    cache = MemoryCache(2, "rubric")
    await cache.put("q", "a", "b", Winner.FIRST)
    await cache.put("q", "a", "c", Winner.FIRST)
    await get(cache, "q", "a", "b")
    await cache.put("q", "a", "d", Winner.TIE)

    assert await get(cache, "q", "a", "c") is None
    assert await get(cache, "q", "a", "b") is Winner.FIRST
    assert await get(cache, "q", "d", "a") is Winner.TIE


async def test_memory_of_size_zero_keeps_nothing():
    cache = MemoryCache(0, "rubric")
    await cache.put("q", "a", "b", Winner.FIRST)

    assert len(cache) == 0


async def test_opens_redis_only_when_a_url_is_set():
    async with open_cache(Settings(_env_file=None), "rubric") as cache:
        assert isinstance(cache, MemoryCache)
    async with open_cache(Settings(_env_file=None, redis_url="redis://localhost:1/0"), "rubric") as cache:
        assert isinstance(cache, RedisCache)
        # Nothing listens there: a miss, quickly, not an error.
        assert await get(cache, "q", "a", "b") is None
