"""Comparisons already paid for. The key is the question (the rubric: model, instructions, tie band, truncation), the
query and the two texts, in a fixed order, so A against B and B against A are one entry.

Two stores share one interface: MemoryCache, per process, and RedisCache, shared by every worker and replica. A Redis
that is down or slow counts as a miss: the comparison is asked again and search still works."""

import hashlib
import logging
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Protocol

import redis.asyncio as redis

from reranker.config import Settings
from reranker.ranking import Winner

log = logging.getLogger(__name__)

_FLIPPED = {Winner.FIRST: Winner.SECOND, Winner.SECOND: Winner.FIRST, Winner.TIE: Winner.TIE}

Pair = tuple[str, str]


class ComparisonCache(Protocol):
    async def get_many(self, query: str, pairs: list[Pair], rubric: str | None = None) -> list[Winner | None]:
        """Each pair's winner (in the pair's own order), or None when it has not been asked. rubric: another
        question than the cache's own (another ranker asks it differently)."""
        ...

    async def put(self, query: str, first: str, second: str, winner: Winner, rubric: str | None = None) -> None: ...


def entry_key(rubric: str, query: str, first: str, second: str) -> tuple[str, bool]:
    """The key of a comparison, and whether the pair is stored the other way round."""
    flipped = second < first
    low, high = (second, first) if flipped else (first, second)
    key = hashlib.sha256("\u0000".join([rubric, query, low, high]).encode()).hexdigest()
    return key, flipped


def _as_stored(winner: Winner, flipped: bool) -> Winner:
    return _FLIPPED[winner] if flipped else winner


class MemoryCache:
    """In this process only, least recently used forgotten beyond size."""

    def __init__(self, size: int, rubric: str):
        self._size = size
        self._rubric = rubric
        self._entries: OrderedDict[str, Winner] = OrderedDict()

    async def get_many(self, query: str, pairs: list[Pair], rubric: str | None = None) -> list[Winner | None]:
        return [self._get(rubric or self._rubric, query, first, second) for first, second in pairs]

    async def put(self, query: str, first: str, second: str, winner: Winner, rubric: str | None = None) -> None:
        if self._size <= 0:
            return
        key, flipped = entry_key(rubric or self._rubric, query, first, second)
        self._entries[key] = _as_stored(winner, flipped)
        self._entries.move_to_end(key)
        while len(self._entries) > self._size:
            self._entries.popitem(last=False)

    def __len__(self) -> int:
        return len(self._entries)

    def _get(self, rubric: str, query: str, first: str, second: str) -> Winner | None:
        key, flipped = entry_key(rubric, query, first, second)
        winner = self._entries.get(key)
        if winner is None:
            return None
        self._entries.move_to_end(key)
        return _as_stored(winner, flipped)


class RedisCache:
    """Shared by every worker and replica. Eviction is Redis's own (run it with a maxmemory policy such as
    allkeys-lru); ttl_seconds, when set, also expires each entry."""

    KEY_PREFIX = "reranker:comparison:"

    def __init__(self, client: redis.Redis, rubric: str, ttl_seconds: int | None = None):
        self._client = client
        self._rubric = rubric
        self._ttl = ttl_seconds or None

    async def get_many(self, query: str, pairs: list[Pair], rubric: str | None = None) -> list[Winner | None]:
        if not pairs:
            return []
        keys = [entry_key(rubric or self._rubric, query, first, second) for first, second in pairs]
        try:
            values = await self._client.mget([self.KEY_PREFIX + key for key, _ in keys])
        except (redis.RedisError, OSError) as error:
            log.warning("comparison cache unavailable, asking again: %s", error)
            return [None] * len(pairs)
        return [self._read(value, flipped) for value, (_, flipped) in zip(values, keys, strict=True)]

    async def put(self, query: str, first: str, second: str, winner: Winner, rubric: str | None = None) -> None:
        key, flipped = entry_key(rubric or self._rubric, query, first, second)
        try:
            await self._client.set(self.KEY_PREFIX + key, _as_stored(winner, flipped).value, ex=self._ttl)
        except (redis.RedisError, OSError) as error:
            log.warning("could not keep a comparison in the cache: %s", error)

    @staticmethod
    def _read(value: bytes | str | None, flipped: bool) -> Winner | None:
        if value is None:
            return None
        try:
            winner = Winner(value.decode() if isinstance(value, bytes) else value)
        except ValueError:
            return None
        return _as_stored(winner, flipped)


@asynccontextmanager
async def open_cache(settings: Settings, rubric: str) -> AsyncIterator[ComparisonCache]:
    """The cache the settings ask for: Redis when redis_url is set, else memory."""
    if not settings.redis_url:
        yield MemoryCache(settings.cache_size, rubric)
        return
    client = redis.Redis.from_url(
        settings.redis_url,
        socket_timeout=settings.redis_timeout_seconds,
        socket_connect_timeout=settings.redis_timeout_seconds,
    )
    try:
        yield RedisCache(client, rubric, settings.cache_ttl_seconds)
    finally:
        await client.aclose()
