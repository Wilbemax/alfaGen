from __future__ import annotations

import pytest

from app.detectors.base import DetectorConfig
from app.detectors.regex_detector import RegexDetector


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
async def test_regex_layer_does_not_detect_address(regex_detector):
    matches = await regex_detector.detect("Адрес: г. Москва, ул. Тверская, 15, кв. 45")
    assert not any(m.entity_type in {"ADDRESS", "ADDRESS_PARTIAL"} for m in matches)


@pytest.mark.asyncio
async def test_no_false_positive_on_plain_text(regex_detector):
    matches = await regex_detector.detect("Это обычный текст без персональных данных.")
    assert len(matches) == 0


@pytest.mark.asyncio
async def test_detect_multiple_entities(regex_detector):
    text = "Иван Иванов, email: test@mail.ru, телефон: +7 999 123-45-67"
    matches = await regex_detector.detect(text)
    assert len(matches) >= 2


@pytest.mark.asyncio
async def test_regex_layer_does_not_detect_place_of_birth(regex_detector):
    matches = await regex_detector.detect("Место рождения: г. Москва")
    assert not any(m.entity_type == "PLACE_OF_BIRTH" for m in matches)


@pytest.mark.asyncio
async def test_regex_layer_does_not_detect_passport_issuer(regex_detector):
    matches = await regex_detector.detect("Орган выдавший паспорт: ОВД района Тверской")
    assert not any(m.entity_type == "PASSPORT_ISSUER" for m in matches)


@pytest.mark.asyncio
async def test_regex_layer_does_not_detect_person(regex_detector):
    matches = await regex_detector.detect("иванов иван иванович")
    assert not any(m.entity_type == "PERSON" for m in matches)


@pytest.mark.asyncio
async def test_detect_date_year_first(regex_detector):
    matches = await regex_detector.detect("Дата рождения: 1990.05.12")
    assert any(m.entity_type == "DATE_OF_BIRTH" for m in matches)


@pytest.mark.asyncio
async def test_detect_date_in_words(regex_detector):
    matches = await regex_detector.detect("Дата рождения: 12 мая 1990 года")
    assert any(m.entity_type == "DATE_OF_BIRTH" for m in matches)


@pytest.mark.asyncio
async def test_historical_person_not_masked(regex_detector):
    matches = await regex_detector.detect("Александр Сергеевич Пушкин родился в 1799 году")
    assert not any(m.entity_type == "PERSON" for m in matches)


@pytest.mark.asyncio
async def test_year_not_detected_as_passport(regex_detector):
    matches = await regex_detector.detect("В 1990 году было 1000 человек")
    assert not any(m.entity_type == "PASSPORT_SERIES" for m in matches)


@pytest.mark.asyncio
async def test_bank_branch_address_not_masked(regex_detector):
    matches = await regex_detector.detect("Отделение Альфа-Банка: г. Москва, ул. Тверская, 10")
    assert not any(m.entity_type in ("ADDRESS", "ADDRESS_PARTIAL") for m in matches)


@pytest.mark.asyncio
async def test_passport_series_number_phrasing(regex_detector):
    # «серия ... номер ...» — слова-метки остаются, обе группы имеют тип PASSPORT
    matches = await regex_detector.detect("Серия 4509 номер 123456")
    passport = [m for m in matches if m.entity_type == "PASSPORT"]
    assert [m.text for m in passport] == ["4509", "123456"]


@pytest.mark.asyncio
async def test_passport_adjacent_one_span(regex_detector):
    # Подряд идущие серия + номер — один составной спан
    matches = await regex_detector.detect("паспорт 4509 123456")
    assert any(m.entity_type == "PASSPORT" for m in matches)
    passport = next(m for m in matches if m.entity_type == "PASSPORT")
    assert passport.text == "4509 123456"


@pytest.mark.asyncio
async def test_detect_driver_license_two_digit_series(regex_detector):
    matches = await regex_detector.detect("Водительское удостоверение 77 123456")
    assert any(m.entity_type == "DRIVER_LICENSE" for m in matches)


@pytest.mark.asyncio
async def test_citizenship_not_person(regex_detector):
    matches = await regex_detector.detect("Гражданство: Российская Федерация")
    assert not any(m.entity_type == "PERSON" for m in matches)


@pytest.mark.asyncio
async def test_cvv_span_is_value_only(regex_detector):
    matches = await regex_detector.detect("CVV: 123")
    cvv = next(m for m in matches if m.entity_type == "CARD_CVV")
    assert cvv.text == "123"


@pytest.mark.asyncio
async def test_pin_span_is_value_only(regex_detector):
    matches = await regex_detector.detect("Пин-код: 4321")
    pin = next(m for m in matches if m.entity_type == "CARD_PIN")
    assert pin.text == "4321"


@pytest.mark.asyncio
async def test_place_of_birth_is_outside_regex_scope(regex_detector):
    matches = await regex_detector.detect("Место рождения: г. Москва")
    assert not any(m.entity_type == "PLACE_OF_BIRTH" for m in matches)


@pytest.mark.asyncio
async def test_passport_issuer_is_outside_regex_scope(regex_detector):
    matches = await regex_detector.detect("Орган выдавший паспорт: ОВД района Тверской")
    assert not any(m.entity_type == "PASSPORT_ISSUER" for m in matches)


@pytest.mark.asyncio
async def test_passport_one_span(regex_detector):
    matches = await regex_detector.detect("паспорт 4509 123456")
    assert any(m.entity_type == "PASSPORT" for m in matches)
    passport = next(m for m in matches if m.entity_type == "PASSPORT")
    assert passport.text == "4509 123456"


@pytest.mark.asyncio
async def test_year_in_date_does_not_cancel_date(regex_detector):
    matches = await regex_detector.detect("Дата рождения: 12.05.1990")
    assert any(m.entity_type == "DATE_OF_BIRTH" for m in matches)
    assert not any(m.entity_type == "PASSPORT_SERIES" for m in matches)


@pytest.mark.asyncio
async def test_card_digits_not_passport_series(regex_detector):
    matches = await regex_detector.detect("Карта 4276 1234 5678 9012")
    assert any(m.entity_type == "BANK_CARD" for m in matches)
    assert not any(m.entity_type == "PASSPORT_SERIES" for m in matches)


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["4509 123456", "4509123456", "45 09 123456"])
async def test_passport_supported_forms_need_context(regex_detector, value):
    text = f"паспорт {value}"
    matches = await regex_detector.detect(text)
    passport = next(m for m in matches if m.entity_type == "PASSPORT")
    assert passport.text == value
    assert text[passport.start:passport.end] == value


@pytest.mark.asyncio
async def test_passport_without_context_is_not_detected(regex_detector):
    matches = await regex_detector.detect("Значение 45 09 123456 осталось прежним")
    assert not any(m.entity_type == "PASSPORT" for m in matches)


@pytest.mark.asyncio
async def test_split_passport_labels_are_outside_spans(regex_detector):
    text = "серия 4509 номер 123456"
    matches = await regex_detector.detect(text)
    document = [m for m in matches if m.entity_type == "PASSPORT"]
    assert [m.text for m in document] == ["4509", "123456"]
    assert all("серия" not in m.text.casefold() and "номер" not in m.text.casefold() for m in document)


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["77 01 123456", "7701 123456", "77 123456"])
async def test_driver_license_supported_forms_with_context(regex_detector, value):
    matches = await regex_detector.detect(f"водительское удостоверение {value}")
    license_match = next(m for m in matches if m.entity_type == "DRIVER_LICENSE")
    assert license_match.text == value


@pytest.mark.asyncio
async def test_short_driver_license_pattern_requires_context(regex_detector):
    matches = await regex_detector.detect("Код 77 123456 используется в отчёте")
    assert not any(m.entity_type == "DRIVER_LICENSE" for m in matches)


@pytest.mark.asyncio
async def test_passport_beats_driver_license_and_inn(regex_detector):
    matches = await regex_detector.detect("паспорт и права 4509123456")
    assert [(m.entity_type, m.text) for m in matches] == [("PASSPORT", "4509123456")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text, expected_type",
    [
        ("Дата рождения 12/05/1990", "DATE_OF_BIRTH"),
        ("Дата рождения 05/12/1990", "DATE_OF_BIRTH"),
        ("Дата рождения 1990-05-12", "DATE_OF_BIRTH"),
        ("Дата рождения 1990.12.05", "DATE_OF_BIRTH"),
        ("Дата рождения 1990.31.12", "DATE_OF_BIRTH"),
        ("родился 12 мая 1990 года", "DATE_OF_BIRTH"),
        ("Паспорт выдан 15.03.2015", "PASSPORT_ISSUE_DATE"),
    ],
)
async def test_date_forms_and_context(regex_detector, text, expected_type):
    matches = await regex_detector.detect(text)
    assert any(m.entity_type == expected_type for m in matches)


@pytest.mark.asyncio
async def test_year_and_counts_are_not_dates_or_documents(regex_detector):
    matches = await regex_detector.detect("в 2024 году было 1500 заявок")
    forbidden = {"DATE_OF_BIRTH", "PASSPORT_ISSUE_DATE", "PASSPORT", "DRIVER_LICENSE"}
    assert not any(m.entity_type in forbidden for m in matches)


@pytest.mark.asyncio
async def test_cvv_and_pin_require_context(regex_detector):
    plain = await regex_detector.detect("Числа 123 и 4321")
    assert not any(m.entity_type in {"CARD_CVV", "CARD_PIN"} for m in plain)
    contextual = await regex_detector.detect("CVV 123, пин-код 4321")
    assert {m.entity_type for m in contextual} >= {"CARD_CVV", "CARD_PIN"}


@pytest.mark.asyncio
async def test_overlap_resolution_happens_before_threshold(regex_detector):
    from app.models.pii import PIIMatch

    matches = [
        PIIMatch("INN", "1234567890", 0, 10, 0.99, "regex"),
        PIIMatch("PASSPORT", "1234567890", 0, 10, 0.49, "regex"),
    ]
    resolved = regex_detector._resolve_overlaps(matches)
    assert [match.entity_type for match in resolved] == ["PASSPORT"]
    assert [match for match in resolved if match.confidence >= regex_detector.config.confidence_threshold] == []
