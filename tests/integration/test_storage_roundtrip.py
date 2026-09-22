from __future__ import annotations

import logging

import pytest
from cryptography.fernet import Fernet

from app.config.settings import settings
from app.core.payload_store import MODE_REDIS, PayloadStore, PayloadStoreError
from app.core.pipeline import Pipeline, PipelineError


class SharedRedis:
    """Общий fake Redis для двух независимых PayloadStore."""

    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}
        self.ttl: dict[str, int | None] = {}
        self.fail_get = False
        self.fail_set = False
        self.get_error: Exception = ConnectionError("get failed")
        self.set_error: Exception = ConnectionError("set failed")
        self.pings = 0

    async def ping(self):
        self.pings += 1
        return True

    async def set(self, key, value, ex=None, nx=False):
        if self.fail_set:
            raise self.set_error
        if nx and key in self.data:
            return None
        self.data[key] = value
        self.ttl[key] = ex
        return True

    async def get(self, key):
        if self.fail_get:
            raise self.get_error
        return self.data.get(key)

    async def aclose(self):
        return None


def _configure_shared(monkeypatch, shared: SharedRedis) -> str:
    key = Fernet.generate_key().decode("utf-8")
    monkeypatch.setattr(settings, "uvicorn_workers", 2)
    monkeypatch.setattr(settings, "payload_store_key", key)
    monkeypatch.setattr(settings, "payload_store_allow_memory_fallback", False)
    monkeypatch.setattr("redis.asyncio.Redis", lambda *args, **kwargs: shared)
    return key


@pytest.mark.asyncio
async def test_two_stores_exact_cross_store_roundtrip(monkeypatch, caplog) -> None:
    """Pipeline A маскирует, Pipeline B с другим store демаскирует ту же Redis-запись."""
    shared = SharedRedis()
    key = _configure_shared(monkeypatch, shared)
    store_a = PayloadStore(ttl_seconds=3600, key_prefix="test:payload:")
    store_b = PayloadStore(ttl_seconds=3600, key_prefix="test:payload:")
    assert store_a is not store_b

    pipeline_a = Pipeline(store=store_a)
    pipeline_b = Pipeline(store=store_b)

    original = "Заявка №42: Иван Петров, почта ivan.petrov@example.com — срочно"
    payload_id = "roundtrip-user@example.com"

    with caplog.at_level(logging.DEBUG):
        masked = await pipeline_a.process(original, payload_id)
        assert masked != original
        assert store_a.mode == MODE_REDIS
        assert len(shared.data) == 1
        ciphertext = next(iter(shared.data.values()))
        redis_key = next(iter(shared.data))
        assert original.encode("utf-8") not in ciphertext
        assert masked.encode("utf-8") not in ciphertext
        assert b"ivan.petrov@example.com" not in ciphertext
        assert payload_id not in redis_key
        assert "checker" not in redis_key
        assert shared.ttl[redis_key] == 3600

        restored = await pipeline_b.process(masked, payload_id)
        assert restored == original
        assert store_b.mode == MODE_REDIS
        assert store_a._fernet is not None
        assert store_b._fernet is not None
        assert store_a._fernet is not store_b._fernet

        retry = await pipeline_b.process(original, payload_id)
        assert retry == masked

        leaked = await pipeline_b.process(masked, payload_id, system_id="crm-system")
        assert leaked != original
        foreign = await store_b.get("crm-system", payload_id)
        if foreign is not None:
            assert foreign.original_text != original
        checker = await store_b.get("checker", payload_id)
        assert checker is not None
        assert checker.original_text == original
        assert checker.masked_text == masked

    rendered = "\n".join(f"{record.getMessage()} {record.__dict__}" for record in caplog.records)
    assert original not in rendered
    assert masked not in rendered
    assert payload_id not in rendered
    assert key not in rendered
    assert "ivan.petrov@example.com" not in rendered


@pytest.mark.asyncio
async def test_runtime_storage_error_becomes_safe_pipeline_error(monkeypatch, caplog) -> None:
    """Сбой Redis в production-режиме становится PipelineError без исходника и ключа."""
    shared = SharedRedis()
    key = _configure_shared(monkeypatch, shared)
    store = PayloadStore(ttl_seconds=3600, key_prefix="test:payload:")
    pipeline = Pipeline(store=store)
    original = "Секретный текст ivan.secret@example.com"

    shared.fail_get = True
    shared.get_error = ConnectionError(f"GET {key} {original}")

    with caplog.at_level(logging.DEBUG), pytest.raises(PipelineError, match=r"^storage_error$") as exc_info:
        await pipeline.process(original, "storage-error-payload")

    assert original not in str(exc_info.value)
    assert key not in str(exc_info.value)
    assert store._records == {}
    assert store.mode == MODE_REDIS
    rendered = "\n".join(f"{record.getMessage()} {record.__dict__}" for record in caplog.records)
    assert original not in rendered
    assert key not in rendered
    assert "GET" not in rendered

    shared.fail_get = False
    shared.fail_set = True
    shared.set_error = ConnectionError(f"SET {original}")
    caplog.clear()
    with caplog.at_level(logging.DEBUG), pytest.raises(PipelineError, match=r"^storage_error$") as put_exc:
        await pipeline.process(original, "storage-error-put")

    assert original not in str(put_exc.value)
    assert key not in str(put_exc.value)
    assert store._records == {}
    rendered = "\n".join(f"{record.getMessage()} {record.__dict__}" for record in caplog.records)
    assert original not in rendered
    assert "SET" not in rendered


@pytest.mark.asyncio
async def test_pipeline_uses_injected_store_not_global(monkeypatch) -> None:
    """Ошибка injected store не читает глобальный payload_store."""
    shared = SharedRedis()
    _configure_shared(monkeypatch, shared)

    class ExplodingStore(PayloadStore):
        async def initialize(self) -> None:
            self.ready = True
            self.mode = MODE_REDIS

        async def get(self, scope: str, payload_id: str):
            raise PayloadStoreError("hidden original and key")

    pipeline = Pipeline(store=ExplodingStore())
    with pytest.raises(PipelineError, match=r"^storage_error$") as exc_info:
        await pipeline.process("текст", "pid")
    assert "hidden original" not in str(exc_info.value)
