from __future__ import annotations

import time

import pytest

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
