from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from app.config.settings import settings

if TYPE_CHECKING:
    import redis.asyncio as aioredis

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Rate limiter с поддержкой Redis (shared state) и in-memory fallback.
    Использует алгоритм Token Bucket.
    """

    def __init__(self) -> None:
        self._redis: aioredis.Redis | None = None
        self.enabled = settings.rate_limit_enabled
        self.default_rps = settings.rate_limit_default_rps
        self.burst = settings.rate_limit_burst
        self.key_prefix = settings.rate_limit_key_prefix
        self._local_buckets: dict[str, tuple[float, float]] = {}  # key -> (tokens, last_refill)

    async def initialize(self) -> None:
        """Инициализация Redis подключения"""
        if not self.enabled:
            return

        try:
            import redis.asyncio as aioredis
            self._redis = aioredis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                password=settings.redis_password,
                socket_timeout=settings.redis_socket_timeout,
                socket_connect_timeout=settings.redis_socket_connect_timeout,
                max_connections=settings.redis_max_connections,
                decode_responses=True,
            )
            # Проверяем подключение
            await self._redis.ping()
            logger.info("Rate limiter initialized with Redis")
        except Exception as e:
            logger.warning(f"Redis unavailable, using in-memory rate limiter: {e}")
            self._redis = None

    async def close(self) -> None:
        if self._redis:
            await self._redis.aclose()
            self._redis = None

    async def check(self, key: str, rps: int | None = None) -> bool:
        """
        Проверяет, разрешен ли запрос.
        Возвращает True если разрешен, False если превышен лимит.
        """
        if not self.enabled:
            return True

        rate = rps or self.default_rps
        bucket_key = f"{self.key_prefix}{key}"

        if self._redis:
            return await self._check_redis(bucket_key, rate)
        else:
            return self._check_local(bucket_key, rate)

    async def _check_redis(self, key: str, rate: int) -> bool:
        """Redis-based token bucket с Lua скриптом для атомарности"""
        assert self._redis is not None

        # Lua скрипт: атомарный token bucket
        lua_script = """
        local key = KEYS[1]
        local rate = tonumber(ARGV[1])
        local burst = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])

        local data = redis.call('HMGET', key, 'tokens', 'last_refill')
        local tokens = tonumber(data[1]) or burst
        local last_refill = tonumber(data[2]) or now

        local elapsed = now - last_refill
        tokens = math.min(burst, tokens + elapsed * rate)

        if tokens >= 1 then
            tokens = tokens - 1
            redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
            redis.call('EXPIRE', key, 60)
            return 1
        else
            redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
            redis.call('EXPIRE', key, 60)
            return 0
        end
        """

        try:
            result = await self._redis.eval(
                lua_script,
                1,
                key,
                rate,
                self.burst,
                time.time(),
            )
            return bool(result)
        except Exception as e:
            logger.warning(f"Redis rate limit error, allowing request: {e}")
            return True

    def _check_local(self, key: str, rate: int) -> bool:
        """In-memory token bucket (для single-instance)"""
        now = time.time()
        tokens, last_refill = self._local_buckets.get(key, (self.burst, now))

        elapsed = now - last_refill
        tokens = min(self.burst, tokens + elapsed * rate)

        if tokens >= 1:
            tokens -= 1
            self._local_buckets[key] = (tokens, now)
            return True
        else:
            self._local_buckets[key] = (tokens, now)
            return False


# Глобальный экземпляр
rate_limiter = RateLimiter()
