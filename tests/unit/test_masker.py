from __future__ import annotations
import pytest

from app.core.masker import Masker
from app.core.tokenizer import TokenGenerator
from app.models.pii import PIIMatch, RequestContext


@pytest.fixture
def masker():
    return Masker()


@pytest.fixture
def context():
    return RequestContext()


def test_mask_single_entity(masker, context):
    text = "Мой email: test@mail.ru"
    entities = [
        PIIMatch(
            entity_type="EMAIL",
            text="test@mail.ru",
            start=11,
            end=23,
            confidence=0.98,
            detector_name="regex",
        )
    ]
    result = masker.mask(text, entities, context)
    assert result.masked_text == "Мой email: [EMAIL_1]"
    assert context.get_original("[EMAIL_1]") == "test@mail.ru"


def test_mask_multiple_entities(masker, context):
    text = "Иван Иванов, email: test@mail.ru, телефон: +7 999 123-45-67"
    entities = [
        PIIMatch(entity_type="PERSON", text="Иван Иванов", start=0, end=11, confidence=0.9, detector_name="natasha"),
        PIIMatch(entity_type="EMAIL", text="test@mail.ru", start=20, end=32, confidence=0.98, detector_name="regex"),
        PIIMatch(entity_type="PHONE", text="+7 999 123-45-67", start=42, end=58, confidence=0.95, detector_name="regex"),
    ]
    result = masker.mask(text, entities, context)
    assert "[PERSON_1]" in result.masked_text
    assert "[EMAIL_1]" in result.masked_text
    assert "[PHONE_1]" in result.masked_text
    assert "test@mail.ru" not in result.masked_text
    assert "+7 999 123-45-67" not in result.masked_text


def test_mask_preserves_order(masker, context):
    text = "Иван Иванов, email: test@mail.ru"
    entities = [
        PIIMatch(entity_type="PERSON", text="Иван Иванов", start=0, end=11, confidence=0.9, detector_name="natasha"),
        PIIMatch(entity_type="EMAIL", text="test@mail.ru", start=20, end=32, confidence=0.98, detector_name="regex"),
    ]
    result = masker.mask(text, entities, context)
    # Порядок должен сохраниться
    assert result.masked_text.index("[PERSON_1]") < result.masked_text.index("[EMAIL_1]")


def test_unmask(masker, context):
    text = "Мой email: test@mail.ru"
    entities = [
        PIIMatch(entity_type="EMAIL", text="test@mail.ru", start=11, end=23, confidence=0.98, detector_name="regex")
    ]
    masked = masker.mask(text, entities, context)
    unmasked = masker.unmask(masked.masked_text, context)
    assert unmasked == text


def test_unmask_multiple_tokens(masker, context):
    text = "Иван Иванов, email: test@mail.ru"
    entities = [
        PIIMatch(entity_type="PERSON", text="Иван Иванов", start=0, end=11, confidence=0.9, detector_name="natasha"),
        PIIMatch(entity_type="EMAIL", text="test@mail.ru", start=20, end=32, confidence=0.98, detector_name="regex"),
    ]
    masked = masker.mask(text, entities, context)
    unmasked = masker.unmask(masked.masked_text, context)
    assert unmasked == text


def test_unmask_llm_response(masker, context):
    """Демаскирование ответа LLM, где токены могут быть в любом месте"""
    text = "Иван Иванов, email: test@mail.ru"
    entities = [
        PIIMatch(entity_type="PERSON", text="Иван Иванов", start=0, end=11, confidence=0.9, detector_name="natasha"),
        PIIMatch(entity_type="EMAIL", text="test@mail.ru", start=19, end=31, confidence=0.98, detector_name="regex"),
    ]
    masker.mask(text, entities, context)

    # LLM ответ с токенами
    llm_response = "Клиент [PERSON_1] с email [EMAIL_1] подтвержден."
    unmasked = masker.unmask(llm_response, context)
    assert "Иван Иванов" in unmasked
    assert "test@mail.ru" in unmasked
    assert "[PERSON_1]" not in unmasked


def test_mask_empty_entities(masker, context):
    text = "Обычный текст"
    result = masker.mask(text, [], context)
    assert result.masked_text == text
    assert result.entities == []


def test_unmask_empty_context(masker, context):
    text = "Текст без токенов"
    assert masker.unmask(text, context) == text


def test_token_generator_unique():
    gen = TokenGenerator()
    t1 = gen.generate("EMAIL")
    t2 = gen.generate("EMAIL")
    assert t1 == "[EMAIL_1]"
    assert t2 == "[EMAIL_2]"
    assert t1 != t2