from __future__ import annotations

import hashlib
import logging

import pytest
from cryptography.fernet import Fernet

from app.config.settings import settings
from app.core.payload_store import (
    MODE_MEMORY,
    MODE_REDIS,
    PayloadRecord,
    PayloadStore,
    PayloadStoreError,
)


def _record(original: str = "Иван Иванов", masked: str = "[PERSON_0]") -> PayloadRecord:
    return PayloadRecord(
        original_text=original,
        masked_text=masked,
        entity_types=["PERSON"],
    )


def _fernet_key() -> str:
    return Fernet.generate_key().decode("utf-8")


class FakeFernet:
    def encrypt(self, plaintext: bytes) -> bytes:
        return b"cipher:" + plaintext

    def decrypt(self, raw: bytes) -> bytes:
        return raw[len(b"cipher:"):]


class FakeRedis:
    """Минимальный фейк redis.asyncio.Redis: ping, SET NX EX, GET, aclose."""

    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}
        self.ttl: dict[str, int | None] = {}
        self.calls: list[tuple] = []
        self.pings = 0
        self.closed = False
        self.fail_ping = False
        self.fail_set = False
        self.fail_get = False
        self.ping_error: Exception = ConnectionError("ping failed")
        self.set_error: Exception = ConnectionError("set failed")
        self.get_error: Exception = ConnectionError("get failed")

    async def set(self, key, value, ex=None, nx=False):
        self.calls.append(("set", key, value, ex, nx))
        if self.fail_set:
            raise self.set_error
        if nx and key in self.data:
            return None
        self.data[key] = value
        self.ttl[key] = ex
        return True

    async def get(self, key):
        self.calls.append(("get", key))
        if self.fail_get:
            raise self.get_error
        return self.data.get(key)

    async def ping(self):
        self.pings += 1
        if self.fail_ping:
            raise self.ping_error
        return True

    async def aclose(self):
        self.closed = True


def _new_store(**kwargs) -> PayloadStore:
    params = {"ttl_seconds": 3600, "key_prefix": "test:payload:"}
    params.update(kwargs)
    return PayloadStore(**params)


def _arm_redis(store: PayloadStore, fake: FakeRedis, fernet=None) -> None:
    store._redis = fake
    store._fernet = fernet if fernet is not None else FakeFernet()
    store.mode = MODE_REDIS
    store.ready = True
    store.redis_available = True


def _patch_redis(monkeypatch, fake: FakeRedis) -> None:
    monkeypatch.setattr("redis.asyncio.Redis", lambda *args, **kwargs: fake)


def _configure(
    monkeypatch,
    *,
    workers: int,
    key: str,
    fallback: bool,
) -> None:
    monkeypatch.setattr(settings, "uvicorn_workers", workers)
    monkeypatch.setattr(settings, "payload_store_key", key)
    monkeypatch.setattr(settings, "payload_store_allow_memory_fallback", fallback)


@pytest.mark.asyncio
async def test_put_get_uses_sha256_key() -> None:
    """Ключ Redis — SHA-256 от scope и payload_id, сырые id в ключе отсутствуют."""
    store = _new_store()
    fake = FakeRedis()
    _arm_redis(store, fake)

    scope = "crm-system"
    payload_id = "ivan@example.com"
    await store.put(scope, payload_id, _record())

    material = scope.encode("utf-8") + b"\x00" + payload_id.encode("utf-8")
    expected_key = f"test:payload:{hashlib.sha256(material).hexdigest()}"
    assert expected_key in fake.data
    assert payload_id not in expected_key
    assert scope not in expected_key
    assert payload_id.encode() not in next(iter(fake.data.values()))

    record = await store.get(scope, payload_id)
    assert record is not None
    assert record.original_text == "Иван Иванов"


@pytest.mark.asyncio
async def test_ciphertext_does_not_contain_original_or_mask() -> None:
    """Зашифрованное значение не содержит исходник и маску в UTF-8."""
    store = _new_store()
    fake = FakeRedis()
    _arm_redis(store, fake, Fernet(Fernet.generate_key()))

    original = "Иван Иванов"
    masked = "[PERSON_0]"
    await store.put("checker", "payload-1", _record(original=original, masked=masked))

    ciphertext = next(iter(fake.data.values()))
    assert original.encode("utf-8") not in ciphertext
    assert masked.encode("utf-8") not in ciphertext


@pytest.mark.asyncio
async def test_nx_put_is_idempotent() -> None:
    """Вторая запись с тем же payload_id игнорируется (SET NX)."""
    store = _new_store()
    fake = FakeRedis()
    _arm_redis(store, fake)

    payload_id = "payload-1"
    await store.put("checker", payload_id, _record(original="первая запись"))
    await store.put("checker", payload_id, _record(original="вторая запись"))

    record = await store.get("checker", payload_id)
    assert record is not None
    assert record.original_text == "первая запись"

    set_calls = [c for c in fake.calls if c[0] == "set"]
    assert set_calls
    assert all(c[4] is True for c in set_calls)


@pytest.mark.asyncio
async def test_ttl_is_passed_exactly() -> None:
    """EX равен настроенному TTL, в том числе значению по умолчанию 3600."""
    store = _new_store(ttl_seconds=3600)
    fake = FakeRedis()
    _arm_redis(store, fake)
    await store.put("checker", "payload-ttl", _record())
    assert fake.ttl[next(iter(fake.data))] == 3600

    custom = _new_store(ttl_seconds=1234)
    custom_fake = FakeRedis()
    _arm_redis(custom, custom_fake)
    await custom.put("checker", "payload-ttl-custom", _record())
    assert custom_fake.ttl[next(iter(custom_fake.data))] == 1234


@pytest.mark.asyncio
async def test_same_payload_id_in_two_scopes_uses_different_keys() -> None:
    """Одинаковый payload_id в разных system scope не делит Redis-ключ."""
    store = _new_store()
    fake = FakeRedis()
    _arm_redis(store, fake)

    payload_id = "shared-payload@example.com"
    await store.put("crm-system", payload_id, _record(original="запись crm"))
    await store.put("loan-scoring", payload_id, _record(original="запись loan"))

    assert len(fake.data) == 2
    for key in fake.data:
        assert payload_id not in key
        assert "crm-system" not in key
        assert "loan-scoring" not in key

    crm = await store.get("crm-system", payload_id)
    loan = await store.get("loan-scoring", payload_id)
    assert crm is not None
    assert crm.original_text == "запись crm"
    assert loan is not None
    assert loan.original_text == "запись loan"


@pytest.mark.asyncio
async def test_multi_worker_valid_key_and_ping_succeeds(monkeypatch) -> None:
    """workers>1, валидный Fernet и успешный PING → encrypted Redis."""
    key = _fernet_key()
    _configure(monkeypatch, workers=2, key=key, fallback=True)
    fake = FakeRedis()
    _patch_redis(monkeypatch, fake)
    store = _new_store()

    await store.initialize()

    assert store.ready is True
    assert store.mode == MODE_REDIS
    assert store.redis_available is True
    assert fake.pings == 1
    assert key not in "".join(str(call) for call in fake.calls)


@pytest.mark.asyncio
async def test_multi_worker_missing_key_raises(monkeypatch) -> None:
    """workers>1 и пустой ключ → fail-fast, флаг fallback игнорируется."""
    _configure(monkeypatch, workers=2, key="", fallback=True)
    store = _new_store()

    with pytest.raises(RuntimeError, match="PAYLOAD_STORE_KEY"):
        await store.initialize()

    assert store.ready is False
    assert store.mode is None
    assert store.redis_available is False


@pytest.mark.asyncio
async def test_multi_worker_invalid_key_raises(monkeypatch) -> None:
    """Невалидный ключ при workers>1 не деградирует в memory и не попадает в ошибку."""
    secret = "not-a-valid-fernet-key"
    _configure(monkeypatch, workers=2, key=secret, fallback=True)

    def forbid_redis(*args, **kwargs):
        raise AssertionError("Redis must not be contacted when the key is invalid")

    monkeypatch.setattr("redis.asyncio.Redis", forbid_redis)
    store = _new_store()

    with pytest.raises(RuntimeError, match="Invalid PAYLOAD_STORE_KEY") as exc_info:
        await store.initialize()

    assert secret not in str(exc_info.value)
    assert store.ready is False
    assert store.mode != MODE_MEMORY
    assert store._records == {}


@pytest.mark.asyncio
async def test_multi_worker_redis_unavailable_raises(monkeypatch, caplog) -> None:
    """Конструктор Redis падает при workers>1 → fail-fast без деталей подключения."""
    key = _fernet_key()
    _configure(monkeypatch, workers=2, key=key, fallback=False)

    def unavailable(*args, **kwargs):
        raise ConnectionError("password=super-secret refused")

    monkeypatch.setattr("redis.asyncio.Redis", unavailable)
    store = _new_store()

    with caplog.at_level(logging.DEBUG), pytest.raises(RuntimeError, match="Redis is unavailable") as exc_info:
        await store.initialize()

    assert "super-secret" not in str(exc_info.value)
    assert "super-secret" not in caplog.text
    assert key not in str(exc_info.value)
    assert key not in caplog.text
    assert store.mode is None
    assert store.ready is False


@pytest.mark.asyncio
async def test_multi_worker_ping_failure_raises(monkeypatch) -> None:
    """Неуспешный PING при workers>1 → fail-fast, клиент закрыт, memory не включается."""
    key = _fernet_key()
    _configure(monkeypatch, workers=2, key=key, fallback=True)
    fake = FakeRedis()
    fake.fail_ping = True
    fake.ping_error = ConnectionError("ping AUTH super-secret")
    _patch_redis(monkeypatch, fake)
    store = _new_store()

    with pytest.raises(RuntimeError, match="Redis is unavailable") as exc_info:
        await store.initialize()

    assert "super-secret" not in str(exc_info.value)
    assert fake.closed is True
    assert store._redis is None
    assert store.mode is None
    assert store.ready is False


@pytest.mark.asyncio
async def test_multi_worker_runtime_put_failure_does_not_write_local(monkeypatch, caplog) -> None:
    """Сбой SET в multi-worker не создаёт process-local запись и не пишет value в лог."""
    key = _fernet_key()
    _configure(monkeypatch, workers=2, key=key, fallback=False)
    fake = FakeRedis()
    _patch_redis(monkeypatch, fake)
    store = _new_store()
    await store.initialize()

    original = "секретный исходник"
    fake.fail_set = True
    fake.set_error = ConnectionError(f"SET payload {original}")

    with caplog.at_level(logging.DEBUG), pytest.raises(PayloadStoreError, match="write failed") as exc_info:
        await store.put("checker", "pid-put", _record(original=original))

    assert original not in str(exc_info.value)
    assert original not in caplog.text
    assert "SET" not in caplog.text
    assert store._records == {}
    assert store.mode == MODE_REDIS


@pytest.mark.asyncio
async def test_multi_worker_runtime_get_failure_does_not_read_local(monkeypatch, caplog) -> None:
    """Сбой GET в multi-worker не читает local stale record и не логирует команду."""
    key = _fernet_key()
    _configure(monkeypatch, workers=2, key=key, fallback=False)
    fake = FakeRedis()
    _patch_redis(monkeypatch, fake)
    store = _new_store()
    await store.initialize()

    stale = "устаревшая локальная запись"
    store._records[store._memory_key("checker", "pid-get")] = _record(original=stale)
    fake.fail_get = True
    fake.get_error = ConnectionError(f"GET pid-get {stale}")

    with caplog.at_level(logging.DEBUG), pytest.raises(PayloadStoreError, match="read failed") as exc_info:
        await store.get("checker", "pid-get")

    assert stale not in str(exc_info.value)
    assert stale not in caplog.text
    assert "GET" not in caplog.text
    assert store.mode == MODE_REDIS


@pytest.mark.asyncio
async def test_redis_miss_does_not_read_local_record() -> None:
    """Промах Redis не подменяется process-local записью."""
    store = _new_store()
    fake = FakeRedis()
    _arm_redis(store, fake, Fernet(Fernet.generate_key()))
    store._records[store._memory_key("checker", "pid")] = _record(original="stale-local")

    assert await store.get("checker", "pid") is None


@pytest.mark.asyncio
async def test_single_worker_explicit_fallback_uses_memory(monkeypatch, caplog) -> None:
    """Один воркер и разрешённый fallback без ключа работают в memory, с одним warning."""
    _configure(monkeypatch, workers=1, key="   ", fallback=True)
    store = _new_store(ttl_seconds=50)

    with caplog.at_level(logging.WARNING):
        await store.initialize()
        await store.initialize()

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert store.mode == MODE_MEMORY
    assert store.ready is True
    assert store.redis_available is False

    await store.put("checker", "pid", _record(original="первая"))
    await store.put("checker", "pid", _record(original="вторая"))
    await store.put("crm-system", "pid", _record(original="другая система"))

    first = await store.get("checker", "pid")
    other = await store.get("crm-system", "pid")
    assert first is not None
    assert first.original_text == "первая"
    assert other is not None
    assert other.original_text == "другая система"


@pytest.mark.asyncio
async def test_single_worker_fallback_disabled_missing_key_raises(monkeypatch) -> None:
    """Запрет fallback при пустом ключе останавливает startup даже для одного воркера."""
    _configure(monkeypatch, workers=1, key="", fallback=False)
    store = _new_store()

    with pytest.raises(RuntimeError, match="memory fallback is disabled"):
        await store.initialize()

    assert store.ready is False
    assert store.mode is None


@pytest.mark.asyncio
async def test_invalid_explicit_key_does_not_degrade_to_memory(monkeypatch) -> None:
    """Явно заданный невалидный ключ — ошибка, а не отсутствие ключа."""
    secret = "definitely-not-fernet"
    _configure(monkeypatch, workers=1, key=secret, fallback=True)
    store = _new_store()

    with pytest.raises(RuntimeError, match="Invalid PAYLOAD_STORE_KEY") as exc_info:
        await store.initialize()

    assert secret not in str(exc_info.value)
    assert store.mode is None
    assert store.ready is False
    assert store.redis_available is False


@pytest.mark.asyncio
async def test_single_worker_redis_down_uses_memory_when_fallback_allowed(monkeypatch, caplog) -> None:
    """Валидный ключ и недоступный Redis при разрешённом fallback → memory, клиент закрыт."""
    _configure(monkeypatch, workers=1, key=_fernet_key(), fallback=True)
    fake = FakeRedis()
    fake.fail_ping = True
    _patch_redis(monkeypatch, fake)
    store = _new_store()

    with caplog.at_level(logging.WARNING):
        await store.initialize()

    assert store.mode == MODE_MEMORY
    assert store._redis is None
    assert store._fernet is None
    assert fake.closed is True
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1
    record = _record(original="локально")
    await store.put("checker", "pid", record)
    loaded = await store.get("checker", "pid")
    assert loaded is not None
    assert loaded.original_text == "локально"


@pytest.mark.asyncio
async def test_single_worker_redis_down_fallback_disabled_raises(monkeypatch) -> None:
    """Недоступный Redis при запрещённом fallback — startup error."""
    _configure(monkeypatch, workers=1, key=_fernet_key(), fallback=False)
    fake = FakeRedis()
    fake.fail_ping = True
    _patch_redis(monkeypatch, fake)
    store = _new_store()

    with pytest.raises(RuntimeError, match="Redis is unavailable"):
        await store.initialize()

    assert store.mode is None
    assert store.ready is False


@pytest.mark.asyncio
async def test_single_worker_valid_key_uses_redis_even_if_fallback_allowed(monkeypatch) -> None:
    """Успешные ключ и Redis фиксируют redis-режим, fallback на запрос не включается."""
    _configure(monkeypatch, workers=1, key=_fernet_key(), fallback=True)
    fake = FakeRedis()
    _patch_redis(monkeypatch, fake)
    store = _new_store()
    await store.initialize()
    assert store.mode == MODE_REDIS

    fake.fail_set = True
    with pytest.raises(PayloadStoreError):
        await store.put("checker", "pid", _record())
    assert store.mode == MODE_REDIS
    assert store._records == {}


@pytest.mark.asyncio
async def test_decrypt_failure_is_controlled_and_ignores_local() -> None:
    """Битый шифротекст даёт PayloadStoreError и не возвращает local stale record."""
    store = _new_store()
    fake = FakeRedis()
    _arm_redis(store, fake, Fernet(Fernet.generate_key()))
    redis_key = store._redis_key("checker", "pid")
    fake.data[redis_key] = b"not-a-fernet-token"
    store._records[store._memory_key("checker", "pid")] = _record(original="stale-plaintext")

    with pytest.raises(PayloadStoreError, match="decrypt") as exc_info:
        await store.get("checker", "pid")

    assert "stale-plaintext" not in str(exc_info.value)
    assert store.mode == MODE_REDIS


@pytest.mark.asyncio
async def test_initialize_is_idempotent(monkeypatch) -> None:
    """Повторный initialize не открывает второй клиент и не пингует Redis снова."""
    _configure(monkeypatch, workers=2, key=_fernet_key(), fallback=False)
    fake = FakeRedis()
    created = {"count": 0}

    def factory(*args, **kwargs):
        created["count"] += 1
        return fake

    monkeypatch.setattr("redis.asyncio.Redis", factory)
    store = _new_store()
    await store.initialize()
    await store.initialize()

    assert created["count"] == 1
    assert fake.pings == 1
    assert store.ready is True


@pytest.mark.asyncio
async def test_close_resets_state(monkeypatch) -> None:
    """close закрывает клиент и сбрасывает ready/mode, после чего initialize можно повторить."""
    _configure(monkeypatch, workers=2, key=_fernet_key(), fallback=False)
    fake = FakeRedis()
    _patch_redis(monkeypatch, fake)
    store = _new_store()
    await store.initialize()
    await store.put("checker", "pid", _record())

    await store.close()

    assert fake.closed is True
    assert store.ready is False
    assert store.mode is None
    assert store.redis_available is False
    assert store._redis is None
    assert store._fernet is None
    assert store._records == {}

    fake.closed = False
    await store.initialize()
    assert store.ready is True
    assert store.mode == MODE_REDIS
    assert fake.pings == 2


@pytest.mark.asyncio
async def test_local_ttl_expires_record(monkeypatch) -> None:
    """В memory-режиме протухшая запись не читается."""
    import time

    _configure(monkeypatch, workers=1, key="", fallback=True)
    store = _new_store(ttl_seconds=5)
    await store.initialize()
    await store.put("checker", "pid", _record(original="живёт"))
    stored = store._records[store._memory_key("checker", "pid")]
    stored.created_at = time.monotonic() - 10

    assert await store.get("checker", "pid") is None


@pytest.mark.asyncio
async def test_multi_worker_without_redis_raises(monkeypatch) -> None:
    """Совместимость: multi-worker без ключа по-прежнему завершается RuntimeError."""
    _configure(monkeypatch, workers=2, key="", fallback=False)
    store = _new_store()
    with pytest.raises(RuntimeError):
        await store.initialize()


@pytest.mark.asyncio
async def test_single_worker_in_memory_allowed(monkeypatch) -> None:
    """Совместимость: один воркер может работать без Redis, если fallback разрешён."""
    _configure(monkeypatch, workers=1, key="", fallback=True)
    store = _new_store()
    await store.initialize()
    assert store.mode == MODE_MEMORY
