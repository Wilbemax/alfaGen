from __future__ import annotations
import pytest

from app.core.pipeline import Pipeline
from app.core.payload_store import payload_store


@pytest.fixture
async def pipeline():
    p = Pipeline()
    await p.initialize()
    yield p
    payload_store.clear()


@pytest.mark.asyncio
async def test_mask_then_unmask(pipeline, sample_text):
    """Маскирование, затем демаскирование по тому же payload_id."""
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
    record = await payload_store.get(payload_id)
    assert record is not None
    assert record.original_text == sample_text
    assert record.masked_text == masked
    assert record.entity_types


@pytest.mark.asyncio
async def test_unknown_payload_masks_again(pipeline, sample_text):
    """Payload, не совпадающий ни с исходником, ни с маской, считается новым маскированием."""
    payload_id = "unit-unknown-1"
    masked = await pipeline.process(sample_text, payload_id)
    # Отправляем произвольную строку — это новое маскирование
    other = "Совершенно другой текст"
    result = await pipeline.process(other, payload_id)
    assert result != masked