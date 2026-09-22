from __future__ import annotations

import sys
import time
from types import ModuleType

import pytest

import app.detectors.natasha_detector as natasha_module
from app.detectors.natasha_detector import NatashaDetector, NerRunner


def _runner(spans: list[tuple[int, int, str, float]]) -> NerRunner:
    def run(text: str) -> list[tuple[int, int, str, float]]:
        return spans

    return run


@pytest.fixture
def person_text() -> str:
    return "Иванов Иван Иванович"


@pytest.mark.asyncio
async def test_person_span(person_text: str) -> None:
    detector = NatashaDetector(ner_runner=_runner([(0, len(person_text), "PER", 0.9)]))
    await detector.initialize()

    matches = await detector.detect(person_text)

    assert len(matches) == 1
    assert matches[0].entity_type == "PERSON"
    assert matches[0].text == person_text
    assert matches[0].start == 0
    assert matches[0].end == len(person_text)
    assert person_text[matches[0].start : matches[0].end] == matches[0].text


@pytest.mark.asyncio
async def test_person_drops_leading_service_word() -> None:
    text = "Поэт Александр Сергеевич Пушкин"
    name = "Александр Сергеевич Пушкин"
    detector = NatashaDetector(ner_runner=_runner([(0, len(text), "PER", 0.9)]))
    await detector.initialize()

    matches = await detector.detect(text)

    assert len(matches) == 1
    assert matches[0].entity_type == "PERSON"
    assert matches[0].text == name
    assert text[matches[0].start : matches[0].end] == name
    assert not matches[0].text.startswith("Поэт")


@pytest.mark.asyncio
async def test_place_of_birth() -> None:
    text = "родился в городе Казань"
    place = "городе Казань"
    start = text.index(place)
    detector = NatashaDetector(
        ner_runner=_runner([(start, start + len(place), "LOC", 0.9)])
    )
    await detector.initialize()

    matches = await detector.detect(text)

    assert len(matches) == 1
    assert matches[0].entity_type == "PLACE_OF_BIRTH"
    assert matches[0].text == place


@pytest.mark.asyncio
async def test_passport_issuer_keeps_keyword_outside() -> None:
    text = "выдан ОУФМС России по г. Москве"
    org = "ОУФМС России по г. Москве"
    detector = NatashaDetector(ner_runner=_runner([(0, len(text), "ORG", 0.9)]))
    await detector.initialize()

    matches = await detector.detect(text)

    assert len(matches) == 1
    assert matches[0].entity_type == "PASSPORT_ISSUER"
    assert matches[0].text == org
    assert text[matches[0].start : matches[0].end] == org


@pytest.mark.asyncio
async def test_occupied_span_is_not_returned(person_text: str) -> None:
    calls: list[str] = []

    def run(text: str) -> list[tuple[int, int, str, float]]:
        calls.append(text)
        return [(0, len(text), "PER", 0.9)]

    detector = NatashaDetector(ner_runner=run)
    await detector.initialize()

    matches = await detector.detect(person_text, occupied=[(0, len(person_text))])

    assert matches == []
    assert calls == []


@pytest.mark.asyncio
async def test_occupied_is_hidden_without_changing_offsets() -> None:
    text = "Иванов Иван, Петров Петр"
    first = "Иванов Иван"
    second = "Петров Петр"
    second_start = text.index(second)
    calls: list[str] = []

    def run(prepared: str) -> list[tuple[int, int, str, float]]:
        calls.append(prepared)
        return [
            (0, len(first), "PER", 0.9),
            (second_start, second_start + len(second), "PER", 0.9),
        ]

    detector = NatashaDetector(ner_runner=run)
    await detector.initialize()

    matches = await detector.detect(
        text,
        occupied=[(-10, len(first)), (100, 200), (5, 3)],
    )

    assert len(calls) == 1
    assert len(calls[0]) == len(text)
    assert calls[0][: len(first)] == " " * len(first)
    assert calls[0][len(first) :] == text[len(first) :]
    assert [(match.text, match.start, match.end) for match in matches] == [
        (second, second_start, second_start + len(second)),
    ]


@pytest.mark.asyncio
async def test_past_deadline_returns_empty(person_text: str) -> None:
    calls: list[str] = []

    def run(text: str) -> list[tuple[int, int, str, float]]:
        calls.append(text)
        return [(0, len(text), "PER", 0.9)]

    detector = NatashaDetector(ner_runner=run)
    await detector.initialize()

    matches = await detector.detect(person_text, deadline_monotonic=time.monotonic() - 1)

    assert matches == []
    assert calls == []


@pytest.mark.asyncio
async def test_card_only_skips_ner() -> None:
    calls: list[str] = []

    def run(text: str) -> list[tuple[int, int, str, float]]:
        calls.append(text)
        return [(0, 4, "PER", 0.9)]

    detector = NatashaDetector(ner_runner=run)
    await detector.initialize()

    matches = await detector.detect("карта 4276 1234 5678 9012")

    assert matches == []
    assert calls == []


@pytest.mark.asyncio
async def test_model_load_failure_returns_empty(person_text: str) -> None:
    detector = NatashaDetector()

    def fail() -> None:
        raise ImportError("natasha")

    detector._load_models = fail  # type: ignore[method-assign]

    matches = await detector.detect(person_text)

    assert matches == []
    assert detector.config.enabled is False


@pytest.mark.asyncio
async def test_initialize_is_idempotent() -> None:
    detector = NatashaDetector()
    load_calls = 0

    def load() -> None:
        nonlocal load_calls
        load_calls += 1
        detector._ner_runner = _runner([])

    detector._load_models = load  # type: ignore[method-assign]

    await detector.initialize()
    await detector.initialize()

    assert load_calls == 1


def test_model_bundle_is_shared_between_instances(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_natasha = ModuleType("natasha")
    load_calls = 0

    class FakeDoc:
        pass

    class FakeSegmenter:
        pass

    def build_tagger() -> object:
        nonlocal load_calls
        load_calls += 1
        return object()

    fake_natasha.Doc = FakeDoc  # type: ignore[attr-defined]
    fake_natasha.Segmenter = FakeSegmenter  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "natasha", fake_natasha)
    monkeypatch.setattr(natasha_module, "_build_tagger", build_tagger)
    natasha_module._load_natasha_models.cache_clear()
    try:
        first = NatashaDetector()
        second = NatashaDetector()

        first._load_models()
        second._load_models()

        assert load_calls == 1
        assert first._segmenter is second._segmenter
        assert first._tagger is second._tagger
        assert first._doc_cls is second._doc_cls
    finally:
        natasha_module._load_natasha_models.cache_clear()


@pytest.mark.asyncio
async def test_initialize_failure_does_not_escape() -> None:
    detector = NatashaDetector()

    def fail() -> None:
        raise RuntimeError("models unavailable")

    detector._load_models = fail  # type: ignore[method-assign]

    await detector.initialize()

    assert detector.config.enabled is False
    assert await detector.detect("Иванов Иван") == []


@pytest.mark.asyncio
async def test_unknown_location_without_context_is_dropped() -> None:
    text = "Москва рядом"
    detector = NatashaDetector(ner_runner=_runner([(0, 6, "LOC", 0.9)]))
    await detector.initialize()

    assert await detector.detect(text) == []


@pytest.mark.asyncio
async def test_address_location_uses_context_and_exact_offsets() -> None:
    text = "улица Пушкина"
    location = "Пушкина"
    start = text.index(location)
    detector = NatashaDetector(
        ner_runner=_runner([(start, start + len(location), "LOC", 0.9)])
    )
    await detector.initialize()

    matches = await detector.detect(text)

    assert len(matches) == 1
    assert matches[0].entity_type == "ADDRESS"
    assert matches[0].text == text[matches[0].start : matches[0].end]
    assert matches[0].text == text


@pytest.mark.asyncio
async def test_structured_address_expands_from_location() -> None:
    text = "проживает: Москва, 125009, г. Москва, ул. Тверская, д. 15, кв. 45"
    address = "Москва, 125009, г. Москва, ул. Тверская, д. 15, кв. 45"
    location = "Москва"
    start = text.index(location)
    detector = NatashaDetector(
        ner_runner=_runner([(start, start + len(location), "LOC", 0.9)])
    )
    await detector.initialize()

    matches = await detector.detect(text)
    addresses = [match for match in matches if match.entity_type == "ADDRESS"]

    assert addresses
    assert all(match.text == text[match.start:match.end] for match in addresses)
    covered = {
        index
        for match in addresses
        for index in range(match.start, match.end)
    }
    expected_start = text.index(address)
    assert covered == set(range(expected_start, len(text)))
    assert all(match.start >= expected_start for match in addresses)


@pytest.mark.asyncio
async def test_structured_address_does_not_overlap_occupied() -> None:
    text = "проживает: Москва, 125009, г. Москва, ул. Тверская, д. 15, кв. 45"
    location = "Москва"
    location_start = text.index(location)
    occupied_start = text.index("125009")
    occupied = (occupied_start, occupied_start + len("125009"))
    detector = NatashaDetector(
        ner_runner=_runner(
            [(location_start, location_start + len(location), "LOC", 0.9)]
        )
    )
    await detector.initialize()

    matches = await detector.detect(text, occupied=[occupied])
    addresses = [match for match in matches if match.entity_type == "ADDRESS"]

    assert addresses
    assert all(match.text == text[match.start:match.end] for match in addresses)
    assert all(
        match.end <= occupied[0] or match.start >= occupied[1]
        for match in addresses
    )
    covered = {
        index
        for match in addresses
        for index in range(match.start, match.end)
    }
    covered.update(range(*occupied))
    assert covered == set(range(text.index(location), len(text)))


@pytest.mark.asyncio
async def test_location_inside_issuer_is_not_expanded_to_whole_phrase() -> None:
    text = "выдан ОУФМС России по г. Москве"
    location = "Москве"
    start = text.index(location)
    detector = NatashaDetector(
        ner_runner=_runner([(start, start + len(location), "LOC", 0.9)])
    )
    await detector.initialize()

    matches = await detector.detect(text)

    assert not any(match.entity_type == "ADDRESS" for match in matches)


@pytest.mark.asyncio
async def test_structured_address_supports_extended_components() -> None:
    text = (
        "проживает: Республика Татарстан, г. Казань, пр-т Победы, д. 10, "
        "корп. 2, стр. 1, кв. 5"
    )
    address = text.removeprefix("проживает: ")
    location = "Казань"
    start = text.index(location)
    detector = NatashaDetector(
        ner_runner=_runner([(start, start + len(location), "LOC", 0.9)])
    )
    await detector.initialize()

    matches = await detector.detect(text)
    addresses = [match for match in matches if match.entity_type == "ADDRESS"]

    assert len(addresses) == 1
    match = addresses[0]
    assert match.text == address
    assert match.start == text.index(address)
    assert match.end == len(text)
    assert match.text == text[match.start:match.end]


@pytest.mark.asyncio
async def test_structured_address_supports_region_and_road_components() -> None:
    text = (
        "проживает: Московская область, Одинцовский район, г. Одинцово, "
        "Можайское шоссе, д. 1"
    )
    address = text.removeprefix("проживает: ")
    location = "Одинцово"
    start = text.index(location)
    detector = NatashaDetector(
        ner_runner=_runner([(start, start + len(location), "LOC", 0.9)])
    )
    await detector.initialize()

    matches = await detector.detect(text)
    addresses = [match for match in matches if match.entity_type == "ADDRESS"]

    assert len(addresses) == 1
    match = addresses[0]
    assert match.text == address
    assert match.start == text.index(address)
    assert match.end == len(text)
    assert match.text == text[match.start:match.end]


@pytest.mark.asyncio
async def test_region_without_address_context_is_dropped() -> None:
    text = "Республика Татарстан отметила праздник"
    location = "Республика Татарстан"
    detector = NatashaDetector(
        ner_runner=_runner([(0, len(location), "LOC", 0.9)])
    )
    await detector.initialize()

    assert await detector.detect(text) == []


@pytest.mark.asyncio
async def test_organization_without_issuer_context_is_dropped() -> None:
    text = "Компания Ромашка"
    detector = NatashaDetector(
        ner_runner=_runner([(0, len(text), "ORG", 0.9)])
    )
    await detector.initialize()

    assert await detector.detect(text) == []
