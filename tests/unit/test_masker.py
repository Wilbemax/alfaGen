from __future__ import annotations

import pytest

from app.core.masker import Masker
from app.models.pii import PIIMatch


@pytest.fixture
def masker():
    return Masker()


def test_mask_single_entity(masker):
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
    result = masker.mask(text, entities)
    assert result.masked_text == "Мой email: ************"
    assert len(result.masked_text) == len(text)
    assert result.spans
    # Демаскирование восстанавливает оригинал
    assert masker.unmask(result.masked_text, result.spans) == text


def test_mask_multiple_entities(masker):
    text = "Иван Иванов, email: test@mail.ru, телефон: +7 999 123-45-67"
    entities = [
        PIIMatch(entity_type="PERSON", text="Иван Иванов", start=0, end=11, confidence=0.9, detector_name="natasha"),
        PIIMatch(entity_type="EMAIL", text="test@mail.ru", start=20, end=32, confidence=0.98, detector_name="regex"),
        PIIMatch(
            entity_type="PHONE",
            text="+7 999 123-45-67",
            start=43,
            end=59,
            confidence=0.95,
            detector_name="regex",
        ),
    ]
    result = masker.mask(text, entities)
    assert result.masked_text == "***********, email: ************, телефон: ****************"
    assert len(result.masked_text) == len(text)
    assert masker.unmask(result.masked_text, result.spans) == text


def test_mask_preserves_order(masker):
    text = "Иван Иванов, email: test@mail.ru"
    entities = [
        PIIMatch(entity_type="PERSON", text="Иван Иванов", start=0, end=11, confidence=0.9, detector_name="natasha"),
        PIIMatch(entity_type="EMAIL", text="test@mail.ru", start=20, end=32, confidence=0.98, detector_name="regex"),
    ]
    result = masker.mask(text, entities)
    # Неперсональный текст сохраняется
    assert ", email: " in result.masked_text
    assert masker.unmask(result.masked_text, result.spans) == text


def test_mask_empty_entities(masker):
    text = "Обычный текст"
    result = masker.mask(text, [])
    assert result.masked_text == text
    assert result.spans == []


def test_unmask_empty_spans(masker):
    text = "Текст без токенов"
    assert masker.unmask(text, []) == text


def test_mask_person_full_span(masker):
    text = "Клиент Иванов Иван Иванович"
    entities = [
        PIIMatch(
            entity_type="PERSON",
            text="Иванов Иван Иванович",
            start=7,
            end=27,
            confidence=0.95,
            detector_name="regex",
        ),
    ]
    result = masker.mask(text, entities)
    assert result.masked_text == "Клиент ********************"
    assert len(result.masked_text) == len(text)
    assert masker.unmask(result.masked_text, result.spans) == text


def test_mask_passport_full_spans(masker):
    text = "паспорт 4509 123456"
    entities = [
        PIIMatch(entity_type="PASSPORT_SERIES", text="4509", start=8, end=12, confidence=0.9, detector_name="regex"),
        PIIMatch(entity_type="PASSPORT_NUMBER", text="123456", start=13, end=19, confidence=0.9, detector_name="regex"),
    ]
    result = masker.mask(text, entities)
    assert result.masked_text == "паспорт **** ******"
    assert len(result.masked_text) == len(text)
    assert masker.unmask(result.masked_text, result.spans) == text


def test_mask_email_full_span(masker):
    text = "email: ivan.ivanov@example.com"
    entities = [
        PIIMatch(
            entity_type="EMAIL",
            text="ivan.ivanov@example.com",
            start=7,
            end=30,
            confidence=0.98,
            detector_name="regex",
        ),
    ]
    result = masker.mask(text, entities)
    assert result.masked_text == "email: ***********************"
    assert len(result.masked_text) == len(text)
    assert masker.unmask(result.masked_text, result.spans) == text


def test_every_character_inside_span_is_hidden(masker):
    text = "до A b-1. после"
    entity = PIIMatch("EMAIL", "A b-1.", 3, 9, 0.9, "regex")
    result = masker.mask(text, [entity])
    assert result.masked_text == "до ****** после"
    assert result.masked_text[:3] == text[:3]
    assert result.masked_text[9:] == text[9:]
    assert len(result.masked_text) == len(text)


def test_invalid_and_overlapping_spans_are_not_applied_twice(masker):
    text = "abcdefghij"
    entities = [
        PIIMatch("EMAIL", "cdef", 2, 6, 0.9, "regex"),
        PIIMatch("PHONE", "efgh", 4, 8, 0.9, "regex"),
        PIIMatch("INN", "wrong", 8, 10, 0.9, "regex"),
    ]
    result = masker.mask(text, entities)
    assert result.masked_text == "ab****ghij"
    assert result.entity_types == ["EMAIL"]
    assert masker.unmask(result.masked_text, result.spans) == text
