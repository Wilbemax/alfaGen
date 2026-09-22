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
    assert "test@mail.ru" not in result.masked_text
    assert result.spans
    # Демаскирование восстанавливает оригинал
    assert masker.unmask(result.masked_text, result.spans) == text


def test_mask_multiple_entities(masker):
    text = "Иван Иванов, email: test@mail.ru, телефон: +7 999 123-45-67"
    entities = [
        PIIMatch(entity_type="PERSON", text="Иван Иванов", start=0, end=11, confidence=0.9, detector_name="natasha"),
        PIIMatch(entity_type="EMAIL", text="test@mail.ru", start=20, end=32, confidence=0.98, detector_name="regex"),
        PIIMatch(entity_type="PHONE", text="+7 999 123-45-67", start=43, end=60, confidence=0.95, detector_name="regex"),
    ]
    result = masker.mask(text, entities)
    assert "test@mail.ru" not in result.masked_text
    assert "+7 999 123-45-67" not in result.masked_text
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


def test_mask_person_initials(masker):
    text = "Клиент Иванов Иван Иванович"
    entities = [
        PIIMatch(entity_type="PERSON", text="Иванов Иван Иванович", start=7, end=27, confidence=0.95, detector_name="regex"),
    ]
    result = masker.mask(text, entities)
    assert "Иванов Иван Иванович" not in result.masked_text
    assert "И. И. И." in result.masked_text
    assert masker.unmask(result.masked_text, result.spans) == text


def test_mask_passport_partial(masker):
    text = "паспорт 4509 123456"
    entities = [
        PIIMatch(entity_type="PASSPORT_SERIES", text="4509", start=8, end=12, confidence=0.9, detector_name="regex"),
        PIIMatch(entity_type="PASSPORT_NUMBER", text="123456", start=13, end=19, confidence=0.9, detector_name="regex"),
    ]
    result = masker.mask(text, entities)
    assert "4509" not in result.masked_text
    assert "123456" not in result.masked_text
    assert masker.unmask(result.masked_text, result.spans) == text


def test_mask_email_partial(masker):
    text = "email: ivan.ivanov@example.com"
    entities = [
        PIIMatch(entity_type="EMAIL", text="ivan.ivanov@example.com", start=7, end=30, confidence=0.98, detector_name="regex"),
    ]
    result = masker.mask(text, entities)
    assert "ivan.ivanov@example.com" not in result.masked_text
    assert "@example.com" in result.masked_text
    assert masker.unmask(result.masked_text, result.spans) == text