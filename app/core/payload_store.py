from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, NoReturn

from cryptography.fernet import Fernet

from app.config.settings import settings

logger = logging.getLogger(__name__)

MODE_REDIS = "redis"
MODE_MEMORY = "memory"
DEFAULT_SCOPE = "checker"


class PayloadStoreError(Exception):
    """Контролируемая ошибка хранилища.

    Текст исключения не содержит ключ шифрования, исходник, маску
    и аргументы Redis-команды.
    """


def normalize_scope(system_id: str | None) -> str:
    """Стабильный system scope. Пустой и отсутствующий id → ``checker``."""
    if system_id is None:
        return DEFAULT_SCOPE
    scope = system_id.strip()
    return scope or DEFAULT_SCOPE


@dataclass(slots=True)
class PayloadRecord:
    """Запись соответствия «исходник ↔ маска» для одного payload_id."""
    original_text: str
    masked_text: str
    entity_types: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.monotonic)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original": self.original_text,
            "mask": self.masked_text,
            "entity_types": self.entity_types,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PayloadRecord:
        return cls(
            original_text=data["original"],
            masked_text=data["mask"],
            entity_types=list(data.get("entity_types", [])),
        )


class PayloadStore:
    """
    Хранилище соответствий по system scope и payload_id.

    Режим выбирается один раз в ``initialize`` и дальше не меняется:

    * ``redis`` — Fernet-шифротекст в Redis (SET NX EX);
    * ``memory`` — process-local fallback, только single-worker и только
      если ``PAYLOAD_STORE_ALLOW_MEMORY_FALLBACK`` явно разрешает его.

    При ``UVICORN_WORKERS > 1`` fallback запрещён независимо от флага:
    нужны валидный Fernet-ключ и успешный Redis PING.
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
        self._redis: Any = None
        self._fernet: Fernet | None = None
        self.redis_available: bool = False
        self.ready: bool = False
        self.mode: str | None = None
        self._records: dict[str, PayloadRecord] = {}
        self._lock = threading.Lock()
        self._inserts_since_evict = 0
        self._evict_interval = 1024

    async def initialize(self) -> None:
        """Идемпотентная инициализация. Порядок фиксирован.

        1. worker count и fallback policy;
        2. ``PAYLOAD_STORE_KEY``;
        3. проверка ключа как Fernet и создание Fernet;
        4. Redis client;
        5. ``PING``;
        6. режим: encrypted Redis, разрешённый memory fallback или fatal error.
        """
        if self.ready:
            return

        workers = settings.uvicorn_workers
        memory_allowed = workers <= 1 and bool(settings.payload_store_allow_memory_fallback)
        key = (settings.payload_store_key or "").strip()

        if not key:
            if not memory_allowed:
                self._fail(
                    "Multi-worker mode requires PAYLOAD_STORE_KEY and Redis"
                    if workers > 1
                    else "PAYLOAD_STORE_KEY is required when memory fallback is disabled"
                )
            self._activate_memory()
            return

        try:
            fernet = Fernet(key.encode("utf-8"))
        except Exception:
            self._fail("Invalid PAYLOAD_STORE_KEY")

        client = None
        try:
            import redis.asyncio as aioredis

            client = aioredis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                password=settings.redis_password,
                socket_timeout=settings.redis_socket_timeout,
                socket_connect_timeout=settings.redis_socket_connect_timeout,
                max_connections=settings.redis_max_connections,
                decode_responses=False,
            )
            await client.ping()
        except Exception:
            if client is not None:
                with suppress(Exception):
                    await client.aclose()
            if not memory_allowed:
                self._fail("Redis is unavailable for the payload store")
            self._activate_memory()
            return

        self._redis = client
        self._fernet = fernet
        self.redis_available = True
        self.mode = MODE_REDIS
        self.ready = True
        logger.info("Payload store initialized with Redis")

    async def close(self) -> None:
        """Закрывает Redis и сбрасывает ready/mode, не трогая секреты в логах."""
        client = self._redis
        self._redis = None
        self._fernet = None
        self.redis_available = False
        self.mode = None
        self.ready = False
        with self._lock:
            self._records.clear()
            self._inserts_since_evict = 0
        if client is not None:
            await client.aclose()

    async def put(self, scope: str, payload_id: str, record: PayloadRecord) -> bool:
        """Сохраняет запись, если ключа ещё нет. False означает, что победила другая запись."""
        if self.mode == MODE_MEMORY:
            return self._put_local(scope, payload_id, record)
        redis, fernet = self._require_redis()
        key = self._redis_key(scope, payload_id)
        try:
            plaintext = json.dumps(record.to_dict(), ensure_ascii=False).encode("utf-8")
            ciphertext = fernet.encrypt(plaintext)
            created = await redis.set(key, ciphertext, ex=int(self._ttl), nx=True)
        except Exception as exc:
            logger.warning("Payload store write failed (%s)", type(exc).__name__)
            raise PayloadStoreError("Payload store write failed") from None
        return bool(created)

    async def get(self, scope: str, payload_id: str) -> PayloadRecord | None:
        """Возвращает запись или None. В Redis-режиме local store не читается."""
        if self.mode == MODE_MEMORY:
            return self._get_local(scope, payload_id)
        redis, fernet = self._require_redis()
        key = self._redis_key(scope, payload_id)
        try:
            raw = await redis.get(key)
        except Exception as exc:
            logger.warning("Payload store read failed (%s)", type(exc).__name__)
            raise PayloadStoreError("Payload store read failed") from None
        if raw is None:
            return None
        try:
            plaintext = fernet.decrypt(raw)
            parsed = json.loads(plaintext.decode("utf-8"))
        except Exception as exc:
            logger.warning("Payload store decrypt failed (%s)", type(exc).__name__)
            raise PayloadStoreError("Payload store record failed to decrypt") from None
        if not isinstance(parsed, dict):
            logger.warning("Payload store decrypt failed (ValueError)")
            raise PayloadStoreError("Payload store record failed to decrypt")
        try:
            return PayloadRecord.from_dict(parsed)
        except Exception as exc:
            logger.warning("Payload store decrypt failed (%s)", type(exc).__name__)
            raise PayloadStoreError("Payload store record failed to decrypt") from None

    def _require_redis(self) -> tuple[Any, Fernet]:
        if (
            self.ready
            and self.mode == MODE_REDIS
            and self._redis is not None
            and self._fernet is not None
        ):
            return self._redis, self._fernet
        raise PayloadStoreError("Payload store is not ready")

    def _fail(self, message: str) -> NoReturn:
        logger.critical(message)
        raise RuntimeError(message) from None

    def _activate_memory(self) -> None:
        self._redis = None
        self._fernet = None
        self.redis_available = False
        self.mode = MODE_MEMORY
        self.ready = True
        logger.warning("Payload store using in-memory fallback for a single worker")

    def _redis_key(self, scope: str, payload_id: str) -> str:
        """SHA-256 от scope и payload_id. Сырые id в ключе Redis не используются."""
        normalized = normalize_scope(scope)
        material = normalized.encode("utf-8") + b"\x00" + payload_id.encode("utf-8")
        digest = hashlib.sha256(material).hexdigest()
        return f"{self._key_prefix}{digest}"

    def _memory_key(self, scope: str, payload_id: str) -> str:
        return normalize_scope(scope) + "\x00" + payload_id

    def _put_local(self, scope: str, payload_id: str, record: PayloadRecord) -> bool:
        memory_key = self._memory_key(scope, payload_id)
        with self._lock:
            # Идемпотентность: если ключ уже есть — не перезаписываем.
            if memory_key in self._records:
                return False
            # Амортизированная очистка: полный проход по записям делаем не чаще,
            # чем раз в _evict_interval вставок. Корректность TTL обеспечивает
            # ленивая проверка в _get_local.
            self._inserts_since_evict += 1
            if self._inserts_since_evict >= self._evict_interval:
                self._evict_expired_locked()
                self._inserts_since_evict = 0
            if len(self._records) >= self._max_entries and memory_key not in self._records:
                self._evict_oldest_locked()
            self._records[memory_key] = record
        return True

    def _get_local(self, scope: str, payload_id: str) -> PayloadRecord | None:
        memory_key = self._memory_key(scope, payload_id)
        with self._lock:
            record = self._records.get(memory_key)
            if record is None:
                return None
            if time.monotonic() - record.created_at > self._ttl:
                del self._records[memory_key]
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
