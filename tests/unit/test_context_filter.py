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
    person = _match("PERSON", "клиент Иванов Иван", 0, 18)
    result = context_filter.filter(text, [person])
    assert len(result) == 1
    assert result[0].text == "Иванов Иван"
    assert result[0].start == 7
    assert result[0].end == 18


def test_service_word_only_person_removed(context_filter: ContextFilter) -> None:
    text = "клиент"
    person = _match("PERSON", "клиент", 0, 6)
    result = context_filter.filter(text, [person])
    assert result == []


def test_overlap_passport_wins_over_address(context_filter: ContextFilter) -> None:
    """ADDRESS [0,10) и PASSPORT [6,16): специфичность PASSPORT выше -> только PASSPORT."""
    text = "адрес паспортные данные"
    address = _match("ADDRESS", "адрес пасп", 0, 10)
    passport = _match("PASSPORT", "паспортные", 6, 16)
    result = context_filter.filter(text, [address, passport])
    assert [m.entity_type for m in result] == ["PASSPORT"]


def test_touching_spans_not_overlap(context_filter: ContextFilter) -> None:
    """Касающиеся границы [0,6) и [6,12) не считаются пересечением."""
    text = "a@b.ruc@d.ru"
    first = _match("EMAIL", "a@b.ru", 0, 6)
    second = _match("EMAIL", "c@d.ru", 6, 12)
    result = context_filter.filter(text, [first, second])
    assert [m.entity_type for m in result] == ["EMAIL", "EMAIL"]


def test_trimming_with_tab_uses_real_offsets(context_filter: ContextFilter) -> None:
    """'клиент\\tИванов Иван' -> start указывает на 'И', без пробелов."""
    text = "клиент\tИванов Иван"
    person = _match("PERSON", "клиент\tИванов Иван", 0, 18)
    result = context_filter.filter(text, [person])
    assert len(result) == 1
    assert result[0].text == "Иванов Иван"
    assert result[0].start == 7
    assert result[0].end == 18
    # Инвариант: new_text == text[new_start:new_end]
    assert result[0].text == text[result[0].start:result[0].end]


def test_trimming_with_multiple_spaces(context_filter: ContextFilter) -> None:
    """Множественные пробелы между словами учитываются при обрезке."""
    text = "клиент   Иванов Иван"
    person = _match("PERSON", "клиент   Иванов Иван", 0, 20)
    result = context_filter.filter(text, [person])
    assert len(result) == 1
    assert result[0].text == "Иванов Иван"
    assert result[0].start == 9
    assert result[0].end == 20
    assert result[0].text == text[result[0].start:result[0].end]


def test_touching_spans_do_not_overlap(context_filter: ContextFilter) -> None:
    text = "a@b.ru8(900)111-22-33"
    email = _match("EMAIL", "a@b.ru", 0, 6)
    phone = _match("PHONE", "8(900)111-22-33", 6, len(text))
    result = context_filter.filter(text, [phone, email])
    assert [(item.entity_type, item.start, item.end) for item in result] == [
        ("EMAIL", 0, 6),
        ("PHONE", 6, len(text)),
    ]


def test_invalid_source_span_is_removed(context_filter: ContextFilter) -> None:
    text = "ИНН 7707083893"
    invalid = _match("INN", "7707083892", 4, 14)
    assert context_filter.filter(text, [invalid]) == []


def test_person_trimming_preserves_tabs_and_multiple_spaces(
    context_filter: ContextFilter,
) -> None:
    text = "клиент\tИванов   Петр"
    person = _match("PERSON", text, 0, len(text))
    result = context_filter.filter(text, [person])
    assert len(result) == 1
    assert result[0].text == "Иванов   Петр"
    assert text[result[0].start:result[0].end] == result[0].text


def test_passport_issuer_wins_only_its_overlap(context_filter: ContextFilter) -> None:
    text = "ОВД района Северный, г. Томск"
    issuer = _match("PASSPORT_ISSUER", "ОВД района Северный", 0, 19)
    address = _match("ADDRESS", text, 0, len(text))
    result = context_filter.filter(text, [address, issuer])
    assert [(item.entity_type, item.text) for item in result] == [
        ("PASSPORT_ISSUER", "ОВД района Северный")
    ]
