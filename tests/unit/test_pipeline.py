from __future__ import annotations
import pytest

from app.core.pipeline import Pipeline
from app.models.request import ProcessRequest, ProcessingMode


@pytest.fixture
async def pipeline():
    p = Pipeline()
    await p.initialize()
    return p


@pytest.mark.asyncio
async def test_detect_only_mode(pipeline, sample_text):
    request = ProcessRequest(
        system_id="crm-system",
        text=sample_text,
        mode=ProcessingMode.DETECT_ONLY,
    )
    result = await pipeline.process(request)
    assert result.entities
    assert any(e.type.value == "EMAIL" for e in result.entities)
    assert any(e.type.value == "PHONE" for e in result.entities)


@pytest.mark.asyncio
async def test_mask_only_mode(pipeline, sample_text):
    request = ProcessRequest(
        system_id="crm-system",
        text=sample_text,
        mode=ProcessingMode.MASK_ONLY,
    )
    result = await pipeline.process(request)
    assert result.masked_text
    assert "test@mail.ru" not in result.masked_text
    assert "[EMAIL_1]" in result.masked_text


@pytest.mark.asyncio
async def test_mask_only_preserves_plain_text(pipeline):
    request = ProcessRequest(
        system_id="crm-system",
        text="Обычный текст без ПДн",
        mode=ProcessingMode.MASK_ONLY,
    )
    result = await pipeline.process(request)
    assert result.masked_text == "Обычный текст без ПДн"
    assert result.entities == []


@pytest.mark.asyncio
async def test_exclusions_historical_persons(pipeline, sample_text_with_exclusions):
    """Исторические личности не должны маскироваться"""
    request = ProcessRequest(
        system_id="crm-system",
        text=sample_text_with_exclusions,
        mode=ProcessingMode.MASK_ONLY,
    )
    result = await pipeline.process(request)
    # Пушкин и Толстой не должны быть замаскированы
    assert "Пушкин" in result.masked_text
    assert "Толстой" in result.masked_text


@pytest.mark.asyncio
async def test_system_specific_rules(pipeline, sample_text):
    """Разные system_id должны иметь разные правила"""
    # crm-system не включает PASSPORT
    request_crm = ProcessRequest(
        system_id="crm-system",
        text=sample_text,
        mode=ProcessingMode.DETECT_ONLY,
    )
    result_crm = await pipeline.process(request_crm)
    crm_types = {e.type.value for e in result_crm.entities}
    assert "PASSPORT_SERIES" not in crm_types

    # loan-scoring включает PASSPORT
    request_loan = ProcessRequest(
        system_id="loan-scoring",
        text=sample_text,
        mode=ProcessingMode.DETECT_ONLY,
    )
    result_loan = await pipeline.process(request_loan)
    loan_types = {e.type.value for e in result_loan.entities}
    assert "PASSPORT_SERIES" in loan_types


@pytest.mark.asyncio
async def test_context_cleared_after_processing(pipeline, sample_text):
    """Контекст должен очищаться после обработки"""
    from app.models.pii import request_context

    request = ProcessRequest(
        system_id="crm-system",
        text=sample_text,
        mode=ProcessingMode.MASK_ONLY,
    )
    await pipeline.process(request)
    assert request_context.get() is None