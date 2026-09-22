import pytest

from app.detectors.context_filter import ContextFilter
from app.models.pii import PIIMatch


@pytest.fixture
def context_filter() -> ContextFilter:
    return ContextFilter()


def _match(
    entity_type: str,
    text: str,
    start: int,
    end: int,
    confidence: float = 0.9,
) -> PIIMatch:
    return PIIMatch(
        entity_type=entity_type,
        text=text,
        start=start,
        end=end,
        confidence=confidence,
        detector_name="test",
    )


def test_bank_card_wins_over_passport_series(context_filter: ContextFilter) -> None:
    text = "карта 4276 1234 5678 9012"
    card = _match("BANK_CARD", "4276 1234 5678 9012", 6, 25)
    series = _match("PASSPORT_SERIES", "4276 1234", 6, 15)
    result = context_filter.filter(text, [series, card])
    assert [m.entity_type for m in result] == ["BANK_CARD"]


def test_longer_and_more_specific_span_wins(context_filter: ContextFilter) -> None:
    text = "паспорт 4509 123456"
    passport = _match("PASSPORT", "4509 123456", 8, 19)
    series = _match("PASSPORT_SERIES", "4509", 8, 12)
    result = context_filter.filter(text, [series, passport])
    assert [m.entity_type for m in result] == ["PASSPORT"]


def test_historical_person_in_span_removed(context_filter: ContextFilter) -> None:
    text = "Александр Сергеевич Пушкин"
    person = _match("PERSON", "Александр Сергеевич Пушкин", 0, 26)
    result = context_filter.filter(text, [person])
    assert result == []


def test_historical_person_nearby_removes_person(context_filter: ContextFilter) -> None:
    text = "Поэт Александр Сергеевич Пушкин написал роман"
    person = _match("PERSON", "Александр Сергеевич Пушкин", 5, 31)
    result = context_filter.filter(text, [person])
    assert result == []


def test_address_near_bank_branch_removed(context_filter: ContextFilter) -> None:
    text = "Отделение Альфа-Банка: г. Москва, ул. Каланчевская, д. 27"
    address = _match("ADDRESS", "г. Москва, ул. Каланчевская, д. 27", 22, 55)
    result = context_filter.filter(text, [address])
    assert result == []


def test_card_holder_kept_with_holder(context_filter: ContextFilter) -> None:
    text = "Держатель карты: IVAN IVANOV"
    holder = _match("CARD_HOLDER", "IVAN IVANOV", 17, 28)
    result = context_filter.filter(text, [holder])
    assert [m.entity_type for m in result] == ["CARD_HOLDER"]


def test_card_holder_removed_without_holder(context_filter: ContextFilter) -> None:
    text = "IVAN IVANOV"
    holder = _match("CARD_HOLDER", "IVAN IVANOV", 0, 11)
    result = context_filter.filter(text, [holder])
    assert result == []


def test_low_confidence_match_removed(context_filter: ContextFilter) -> None:
    text = "ИНН 7707083893"
    inn = _match("INN", "7707083893", 4, 14, confidence=0.49)
    result = context_filter.filter(text, [inn])
    assert result == []


def test_service_word_trimmed_from_person(context_filter: ContextFilter) -> None:
    text = "клиент Иванов Иван"
    person = _match("PERSON", "клиент Иванов Иван", 0, 17)
    result = context_filter.filter(text, [person])
    assert len(result) == 1
    assert result[0].text == "Иванов Иван"
    assert result[0].start == 7
    assert result[0].end == 17


def test_service_word_only_person_removed(context_filter: ContextFilter) -> None:
    text = "клиент"
    person = _match("PERSON", "клиент", 0, 6)
    result = context_filter.filter(text, [person])
    assert result == []