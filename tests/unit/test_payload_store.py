from __future__ import annotations

import hashlib

import pytest

from app.core.payload_store import PayloadRecord, PayloadStore


@pytest.fixture
def store() -> PayloadStore:
    return PayloadStore(ttl_seconds=3600, key_prefix="test:payload:")


def _record(original: str = "Иван Иванов") -> PayloadRecord:
    return PayloadRecord(
        original_text=original,
        masked_text="[PERSON_0]",
        entity_types=["PERSON"],
    )


class FakeRedis:
    """Минимальный фейк redis.asyncio.Redis для тестов."""

    def __init__(self) -> None:
        self.data: dict[bytes, bytes] = {}
        self.calls: list[tuple] = []

    async def set(self, key, value, ex=None, nx=False):
        self.calls.append(("set", key, value, ex, nx))
        if nx and key in self.data:
            return False
        self.data[key] = value
        return True

    async def get(self, key):
        self.calls.append(("get", key))
        return self.data.get(key)

    async def ping(self):
        return True

    async def aclose(self):
        return None


@pytest.mark.asyncio
async def test_put_get_uses_sha256_key(monkeypatch) -> None:
    """Ключ Redis — SHA-256 от payload_id, raw payload_id в ключе отсутствует."""
    store = PayloadStore(ttl_seconds=3600, key_prefix="test:payload:")
    fake = FakeRedis()
    store._redis = fake
    store._fernet = object()  # не None, чтобы включить Redis-ветку

    # Подменяем Fernet на заглушку, чтобы не зависеть от реального ключа.
    class FakeFernet:
        def encrypt(self, plaintext: bytes) -> bytes:
            return b"cipher:" + plaintext

        def decrypt(self, raw: bytes) -> bytes:
            return raw[len(b"cipher:"):]

    store._fernet = FakeFernet()

    payload_id = "ivan@example.com"
    await store.put(payload_id, _record())

    expected_key = f"test:payload:{hashlib.sha256(payload_id.encode('utf-8')).hexdigest()}"
    assert expected_key in fake.data
    assert payload_id.encode() not in fake.data

    record = await store.get(payload_id)
    assert record is not None
    assert record.original_text == "Иван Иванов"


@pytest.mark.asyncio
async def test_ciphertext_does_not_contain_original_text(monkeypatch) -> None:
    """Зашифрованное значение не содержит исходный текст."""
    from cryptography.fernet import Fernet

    store = PayloadStore(ttl_seconds=3600, key_prefix="test:payload:")
    fake = FakeRedis()
    store._redis = fake
    store._fernet = Fernet(Fernet.generate_key())

    original = "Иван Иванов"
    await store.put("payload-1", _record(original=original))

    key = f"test:payload:{hashlib.sha256(b'payload-1').hexdigest()}"
    ciphertext = fake.data[key]
    assert original.encode("utf-8") not in ciphertext


@pytest.mark.asyncio
async def test_nx_put_is_idempotent(monkeypatch) -> None:
    """Вторая запись с тем же payload_id игнорируется (put-if-absent)."""
    store = PayloadStore(ttl_seconds=3600, key_prefix="test:payload:")
    fake = FakeRedis()
    store._redis = fake

    class FakeFernet:
        def encrypt(self, plaintext: bytes) -> bytes:
            return b"cipher:" + plaintext

        def decrypt(self, raw: bytes) -> bytes:
            return raw[len(b"cipher:"):]

    store._fernet = FakeFernet()

    payload_id = "payload-1"
    await store.put(payload_id, _record(original="первая запись"))
    await store.put(payload_id, _record(original="вторая запись"))

    record = await store.get(payload_id)
    assert record is not None
    assert record.original_text == "первая запись"

    # Проверяем, что nx=True передавался в Redis.
    set_calls = [c for c in fake.calls if c[0] == "set"]
    assert set_calls
    assert all(c[4] is True for c in set_calls)


@pytest.mark.asyncio
async def test_multi_worker_without_redis_raises(monkeypatch) -> None:
    """Multi-worker без Redis/Fernet запрещён — RuntimeError."""
    from app.config.settings import settings

    monkeypatch.setattr(settings, "uvicorn_workers", 2)

    store = PayloadStore(ttl_seconds=3600, key_prefix="test:payload:")
    store._redis = None
    store._fernet = None

    with pytest.raises(RuntimeError):
        await store.initialize()


@pytest.mark.asyncio
async def test_single_worker_in_memory_allowed(monkeypatch) -> None:
    """Для одного воркера in-memory fallback разрешён."""
    from app.config.settings import settings

    monkeypatch.setattr(settings, "uvicorn_workers", 1)

    store = PayloadStore(ttl_seconds=3600, key_prefix="test:payload:")
    store._redis = None
    store._fernet = None

    # Не должно бросить исключение.
    await store.initialize()
