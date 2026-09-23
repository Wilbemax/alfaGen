from __future__ import annotations

import asyncio
import time

import pytest

from app.config.settings import settings
from app.core.payload_store import MODE_MEMORY, PayloadStore, payload_store
from app.core.pipeline import Pipeline, PipelineError
from app.utils.tokenizer import count_tokens


@pytest.fixture
async def pipeline():
    p = Pipeline()
    await p.initialize()
    yield p
    payload_store.clear()


@pytest.mark.asyncio
async def test_mask_then_unmask(pipeline, sample_text):
    """Маскирование, затем демаскирование по тому же payload_id (без system_id)."""
    payload_id = "unit-mask-unmask-1"
    masked = await pipeline.process(sample_text, payload_id)
    assert "test@mail.ru" not in masked
    assert "770123456789" not in masked

    unmasked = await pipeline.process(masked, payload_id)
    assert unmasked == sample_text


@pytest.mark.asyncio
async def test_mask_retry_returns_same_mask(pipeline, sample_text):
    """Ретрай маскирования с тем же исходником возвращает ту же маску без пересчёта."""
    payload_id = "unit-retry-1"
    masked1 = await pipeline.process(sample_text, payload_id)
    masked2 = await pipeline.process(sample_text, payload_id)
    assert masked1 == masked2


@pytest.mark.asyncio
async def test_mask_preserves_plain_text(pipeline):
    """Без ПДн маска совпадает с исходником, оба шага возвращают ту же строку."""
    payload_id = "unit-plain-1"
    text = "Обычный текст без ПДн"
    result1 = await pipeline.process(text, payload_id)
    assert result1 == text
    result2 = await pipeline.process(text, payload_id)
    assert result2 == text


@pytest.mark.asyncio
async def test_new_payload_id_masks(pipeline, sample_text):
    """Новый payload_id всегда маскирует."""
    masked1 = await pipeline.process(sample_text, "unit-new-1")
    masked2 = await pipeline.process(sample_text, "unit-new-2")
    assert masked1 == masked2


@pytest.mark.asyncio
async def test_unmask_restores_original(pipeline, sample_text):
    """Демаскирование возвращает исходную строку."""
    payload_id = "unit-restore-1"
    masked = await pipeline.process(sample_text, payload_id)
    restored = await pipeline.process(masked, payload_id)
    assert restored == sample_text


@pytest.mark.asyncio
async def test_store_keeps_record(pipeline, sample_text):
    """Запись хранит исходник, маску и типы сущностей."""
    payload_id = "unit-record-1"
    masked = await pipeline.process(sample_text, payload_id)
    record = await payload_store.get("checker", payload_id)
    assert record is not None
    assert record.original_text == sample_text
    assert record.masked_text == masked
    assert record.entity_types


@pytest.mark.asyncio
async def test_foreign_text_returns_saved_mask(pipeline, sample_text):
    """Чужой текст с известным payload_id возвращает сохранённую маску, не перезаписывая пару."""
    payload_id = "unit-foreign-1"
    masked = await pipeline.process(sample_text, payload_id)
    other = "Совершенно другой текст"
    result = await pipeline.process(other, payload_id)
    assert result == masked
    # Сохранённая пара не перезаписана
    record = await payload_store.get("checker", payload_id)
    assert record is not None
    assert record.original_text == sample_text
    assert record.masked_text == masked


@pytest.mark.asyncio
async def test_demask_disabled_returns_mask(pipeline, sample_text, monkeypatch):
    """При demask_enabled=false второй запрос с маской возвращает маску, не исходник."""
    profile = {
        "enabled": True,
        "demask_enabled": False,
        "enabled_entity_types": None,
    }
    monkeypatch.setattr(
        pipeline,
        "_get_profile",
        lambda system_id: profile if system_id == "support-chat" else pipeline._get_profile(system_id),
    )

    payload_id = "unit-nounmask-1"
    masked = await pipeline.process(sample_text, payload_id, system_id="support-chat")
    second = await pipeline.process(masked, payload_id, system_id="support-chat")
    assert second == masked
    assert second != sample_text


@pytest.mark.asyncio
async def test_unmask_allowed_without_system_id(pipeline, sample_text):
    """Без system_id демаскирование разрешено."""
    payload_id = "unit-nosys-1"
    masked = await pipeline.process(sample_text, payload_id)
    restored = await pipeline.process(masked, payload_id)
    assert restored == sample_text


@pytest.mark.asyncio
async def test_entity_limit_exceeded_raises(pipeline, monkeypatch):
    """Превышение pipeline_max_entities_per_request -> PipelineError."""
    monkeypatch.setattr(settings, "pipeline_max_entities_per_request", 1)

    class FakeCascade:
        async def detect(self, payload, allowed_types=None):
            from app.models.pii import PIIMatch
            return [
                PIIMatch("PERSON", "Иван", 0, 4, 0.9, "test"),
                PIIMatch("EMAIL", "a@b.ru", 5, 11, 0.9, "test"),
            ]

    pipeline.cascade = FakeCascade()
    with pytest.raises(PipelineError) as exc_info:
        await pipeline.process("Иван a@b.ru", "unit-limit-1")
    assert "entity_limit_exceeded" in str(exc_info.value)


@pytest.mark.asyncio
async def test_payload_id_email_not_in_logs(pipeline, sample_text, caplog):
    """Raw payload_id (email) не попадает в логи."""
    import logging

    payload_id = "ivan@example.com"
    with caplog.at_level(logging.INFO):
        await pipeline.process(sample_text, payload_id)

    assert "ivan@example.com" not in caplog.text


class _GateStore(PayloadStore):
    """Два первых чтения ждут друг друга, чтобы оба увидели пустое хранилище."""

    def __init__(self) -> None:
        super().__init__(ttl_seconds=50)
        self.mode = MODE_MEMORY
        self.ready = True
        self._arrived = 0
        self._release = asyncio.Event()

    async def get(self, scope: str, payload_id: str):
        self._arrived += 1
        if self._arrived == 2:
            self._release.set()
        if self._arrived <= 2:
            await self._release.wait()
        return await super().get(scope, payload_id)


@pytest.mark.asyncio
async def test_concurrent_put_returns_winner_mask() -> None:
    """Проигравший SET NX не отдаёт собственную маску."""
    store = _GateStore()

    class _EmptyCascade:
        async def detect(self, payload: str, allowed_types=None):
            return []

    first = Pipeline(store=store)
    second = Pipeline(store=store)
    for item in (first, second):
        item._initialized = True
        item.cascade = _EmptyCascade()

    left, right = await asyncio.gather(
        first.process("первая строка", "same-id"),
        second.process("вторая строка", "same-id"),
    )
    stored = await store.get("checker", "same-id")
    assert stored is not None
    assert left == right == stored.masked_text
    assert {left, stored.original_text} <= {"первая строка", "вторая строка"}


@pytest.mark.asyncio
async def test_close_allows_next_initialize(monkeypatch) -> None:
    monkeypatch.setattr(settings, "uvicorn_workers", 1)
    monkeypatch.setattr(settings, "payload_store_key", "")
    monkeypatch.setattr(settings, "payload_store_allow_memory_fallback", True)
    store = PayloadStore()
    item = Pipeline(store=store)
    await item.initialize()
    await item.close()
    assert item._initialized is False
    assert store.ready is False
    await item.initialize()
    assert item._initialized is True
    assert store.ready is True
    assert store.mode == MODE_MEMORY


@pytest.mark.asyncio
async def test_hundred_thousand_tokens_mask_under_one_second(pipeline) -> None:
    """Длинный текст с несколькими десятками ПДн маскируется быстрее 1 с."""
    phrase = "Клиент Иванов Иван Иванович, паспорт 4509 123456. "
    text = phrase * 30 + ("заявка " * 95_000)
    assert count_tokens(text) >= 90_000
    started = time.perf_counter()
    masked = await pipeline.process(text, "unit-large-1")
    assert time.perf_counter() - started < 1.0
    assert "Иванов Иван Иванович" not in masked
    assert "4509 123456" not in masked
    restored = await pipeline.process(masked, "unit-large-1")
    assert restored == text
