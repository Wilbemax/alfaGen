from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field

from app.config.settings import settings

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PayloadRecord:
    """Запись соответствия «исходник ↔ маска» для одного payload_id."""
    original_text: str
    masked_text: str
    entity_types: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.monotonic)

    def to_dict(self) -> dict:
        return {
            "original": self.original_text,
            "mask": self.masked_text,
            "entity_types": self.entity_types,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PayloadRecord":
        return cls(
            original_text=data["original"],
            masked_text=data["mask"],
            entity_types=list(data.get("entity_types", [])),
        )


class PayloadStore:
    """
    Хранилище соответствий по payload_id.

    Приоритет — Redis (shared state между воркерами), при недоступности —
    in-memory fallback в процессе. Записи протухают по TTL.

    В значении хранятся исходник, маска и типы сущностей. Эти строки
    никогда не попадают в логи.
    """

    def __init__(
        self,
        ttl_seconds: float | None = None,
        max_entries: int = 1_000_000,
        key_prefix: str | None = None,
    ) -> None:
        self._ttl = ttl_seconds if ttl_seconds is not None else settings.payload_store_ttl_seconds
        self._max_entries = max_entries
        self._key_prefix = key_prefix if key_prefix is not None else settings.payload_store_key_prefix
        self._redis = None
        self._records: dict[str, PayloadRecord] = {}
        self._lock = threading.Lock()
        self._inserts_since_evict = 0
        self._evict_interval = 1024

    async def initialize(self) -> None:
        """Инициализация Redis подключения (с fallback на in-memory)."""
        try:
            import redis.asyncio as aioredis
            self._redis = aioredis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                password=settings.redis_password,
                socket_timeout=settings.redis_socket_timeout,
                socket_connect_timeout=settings.redis_socket_connect_timeout,
                decode_responses=True,
            )
            await self._redis.ping()
            logger.info("Payload store initialized with Redis")
        except Exception as e:
            logger.warning(f"Redis unavailable for payload store, using in-memory: {e}")
            self._redis = None

    async def close(self) -> None:
        if self._redis:
            await self._redis.aclose()
            self._redis = None

    async def put(self, payload_id: str, record: PayloadRecord) -> None:
        """Сохраняет запись соответствия по payload_id."""
        if self._redis is not None:
            try:
                await self._redis.set(
                    self._key_prefix + payload_id,
                    json.dumps(record.to_dict(), ensure_ascii=False),
                    ex=int(self._ttl),
                )
                return
            except Exception as e:
                logger.warning(f"Redis payload store put failed, falling back to memory: {e}")
        self._put_local(payload_id, record)

    async def get(self, payload_id: str) -> PayloadRecord | None:
        """Возвращает запись соответствия или None, если её нет/протухла."""
        if self._redis is not None:
            try:
                raw = await self._redis.get(self._key_prefix + payload_id)
                if raw is not None:
                    return PayloadRecord.from_dict(json.loads(raw))
            except Exception as e:
                logger.warning(f"Redis payload store get failed, falling back to memory: {e}")
        return self._get_local(payload_id)

    def _put_local(self, payload_id: str, record: PayloadRecord) -> None:
        with self._lock:
            # Амортизированная очистка: полный проход по записям делаем не чаще,
            # чем раз в _evict_interval вставок. Корректность TTL обеспечивает
            # ленивая проверка в _get_local.
            self._inserts_since_evict += 1
            if self._inserts_since_evict >= self._evict_interval:
                self._evict_expired_locked()
                self._inserts_since_evict = 0
            if len(self._records) >= self._max_entries and payload_id not in self._records:
                self._evict_oldest_locked()
            self._records[payload_id] = record

    def _get_local(self, payload_id: str) -> PayloadRecord | None:
        with self._lock:
            record = self._records.get(payload_id)
            if record is None:
                return None
            if time.monotonic() - record.created_at > self._ttl:
                del self._records[payload_id]
                return None
            return record

    def _evict_expired_locked(self) -> None:
        now = time.monotonic()
        expired = [
            pid for pid, rec in self._records.items()
            if now - rec.created_at > self._ttl
        ]
        for pid in expired:
            del self._records[pid]

    def _evict_oldest_locked(self) -> None:
        if not self._records:
            return
        oldest_id = min(self._records, key=lambda pid: self._records[pid].created_at)
        del self._records[oldest_id]

    def clear(self) -> None:
        with self._lock:
            self._records.clear()


# Глобальный экземпляр хранилища
payload_store = PayloadStore()