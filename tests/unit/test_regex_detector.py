from __future__ import annotations
import pytest

from app.detectors.regex_detector import RegexDetector
from app.detectors.base import DetectorConfig


@pytest.fixture
async def regex_detector():
    detector = RegexDetector(DetectorConfig(enabled=True, confidence_threshold=0.5))
    await detector.initialize()
    return detector


@pytest.mark.asyncio
async def test_detect_email(regex_detector):
    matches = await regex_detector.detect("Мой email: ivan.ivanov@example.com")
    assert len(matches) == 1
    assert matches[0].entity_type == "EMAIL"
    assert matches[0].text == "ivan.ivanov@example.com"


@pytest.mark.asyncio
async def test_detect_phone(regex_detector):
    matches = await regex_detector.detect("Позвоните по номеру +7 (912) 345-67-89")
    assert len(matches) == 1
    assert matches[0].entity_type == "PHONE"
    assert matches[0].text == "+7 (912) 345-67-89"


@pytest.mark.asyncio
async def test_detect_inn(regex_detector):
    matches = await regex_detector.detect("Мой ИНН: 770123456789")
    assert len(matches) == 1
    assert matches[0].entity_type == "INN"
    assert matches[0].text == "770123456789"


@pytest.mark.asyncio
async def test_detect_bank_card(regex_detector):
    matches = await regex_detector.detect("Карта: 4276 1234 5678 9012")
    assert any(m.entity_type == "BANK_CARD" for m in matches)
    bank_card = next(m for m in matches if m.entity_type == "BANK_CARD")
    assert bank_card.text == "4276 1234 5678 9012"


@pytest.mark.asyncio
async def test_detect_passport_dept_code(regex_detector):
    matches = await regex_detector.detect("Код подразделения: 770-123")
    assert len(matches) == 1
    assert matches[0].entity_type == "PASSPORT_DEPT_CODE"
    assert matches[0].text == "770-123"


@pytest.mark.asyncio
async def test_detect_date_of_birth(regex_detector):
    matches = await regex_detector.detect("Дата рождения: 12.05.1990")
    assert len(matches) >= 1
    assert any(m.entity_type == "DATE_OF_BIRTH" for m in matches)


@pytest.mark.asyncio
async def test_detect_cvv(regex_detector):
    matches = await regex_detector.detect("CVV: 123")
    assert len(matches) == 1
    assert matches[0].entity_type == "CARD_CVV"


@pytest.mark.asyncio
async def test_detect_pin(regex_detector):
    matches = await regex_detector.detect("Пин-код: 4321")
    assert any(m.entity_type == "CARD_PIN" for m in matches)


@pytest.mark.asyncio
async def test_detect_address(regex_detector):
    matches = await regex_detector.detect("Адрес: г. Москва, ул. Тверская, 15, кв. 45")
    assert len(matches) >= 1
    assert any(m.entity_type == "ADDRESS" for m in matches)


@pytest.mark.asyncio
async def test_no_false_positive_on_plain_text(regex_detector):
    matches = await regex_detector.detect("Это обычный текст без персональных данных.")
    assert len(matches) == 0


@pytest.mark.asyncio
async def test_detect_multiple_entities(regex_detector):
    text = "Иван Иванов, email: test@mail.ru, телефон: +7 999 123-45-67"
    matches = await regex_detector.detect(text)
    assert len(matches) >= 2