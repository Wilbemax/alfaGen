import pytest

from app.detectors.cascade import CascadeDetector
from app.models.pii import PIIMatch


class FakeRegexDetector:
    def __init__(self, matches: list[PIIMatch]) -> None:
        self._matches = matches

    async def initialize(self) -> None:
        return None

    async def detect(self, text: str) -> list[PIIMatch]:
        return self._matches


class FakeNatashaDetector:
    def __init__(self, matches: list[PIIMatch] | None = None, fail_init: bool = False) -> None:
        self._matches = matches or []
        self.fail_init = fail_init
        self.occupied: list[tuple[int, int]] | None = None
        self.deadline: float | None = None

    async def initialize(self) -> None:
        if self.fail_init:
            raise RuntimeError("init failed")

    async def detect(
        self,
        text: str,
        occupied: list[tuple[int, int]] | None = None,
        deadline_monotonic: float | None = None,
    ) -> list[PIIMatch]:
        self.occupied = occupied
        self.deadline = deadline_monotonic
        return self._matches


class RaisingNatashaDetector(FakeNatashaDetector):
    async def detect(
        self,
        text: str,
        occupied: list[tuple[int, int]] | None = None,
        deadline_monotonic: float | None = None,
    ) -> list[PIIMatch]:
        raise RuntimeError("detect failed")


def _match(entity_type: str, text: str, start: int, end: int) -> PIIMatch:
    return PIIMatch(
        entity_type=entity_type,
        text=text,
        start=start,
        end=end,
        confidence=0.9,
        detector_name="fake",
    )


@pytest.mark.asyncio
async def test_cascade_calls_natasha_with_occupied() -> None:
    email = _match("EMAIL", "ivan@example.com", 0, 16)
    person = _match("PERSON", "Иванов Иван", 17, 28)
    regex = FakeRegexDetector([email])
    natasha = FakeNatashaDetector([person])
    cascade = CascadeDetector(regex_detector=regex, natasha_detector=natasha)
    await cascade.initialize()

    result = await cascade.detect("ivan@example.com Иванов Иван")

    assert natasha.occupied == [(0, 16)]
    assert natasha.deadline is not None
    assert {m.entity_type for m in result} == {"EMAIL", "PERSON"}


@pytest.mark.asyncio
async def test_cascade_survives_natasha_exception() -> None:
    email = _match("EMAIL", "ivan@example.com", 0, 16)
    regex = FakeRegexDetector([email])
    natasha = RaisingNatashaDetector()
    cascade = CascadeDetector(regex_detector=regex, natasha_detector=natasha)
    await cascade.initialize()

    result = await cascade.detect("ivan@example.com")

    assert [m.entity_type for m in result] == ["EMAIL"]


@pytest.mark.asyncio
async def test_allowed_types_filters_results() -> None:
    email = _match("EMAIL", "ivan@example.com", 0, 16)
    person = _match("PERSON", "Иванов Иван", 17, 28)
    regex = FakeRegexDetector([email])
    natasha = FakeNatashaDetector([person])
    cascade = CascadeDetector(regex_detector=regex, natasha_detector=natasha)
    await cascade.initialize()

    result = await cascade.detect("ivan@example.com Иванов Иван", allowed_types={"PERSON"})

    assert [m.entity_type for m in result] == ["PERSON"]


@pytest.mark.asyncio
async def test_empty_allowed_types_returns_empty() -> None:
    email = _match("EMAIL", "ivan@example.com", 0, 16)
    regex = FakeRegexDetector([email])
    natasha = FakeNatashaDetector()
    cascade = CascadeDetector(regex_detector=regex, natasha_detector=natasha)
    await cascade.initialize()

    result = await cascade.detect("ivan@example.com", allowed_types=set())

    assert result == []


@pytest.mark.asyncio
async def test_cascade_without_natasha_returns_regex_only() -> None:
    email = _match("EMAIL", "ivan@example.com", 0, 16)
    regex = FakeRegexDetector([email])
    natasha = FakeNatashaDetector(fail_init=True)
    cascade = CascadeDetector(regex_detector=regex, natasha_detector=natasha)
    await cascade.initialize()

    assert cascade.natasha_available is False

    result = await cascade.detect("ivan@example.com")

    assert [m.entity_type for m in result] == ["EMAIL"]


@pytest.mark.asyncio
async def test_long_text_still_calls_natasha() -> None:
    regex = FakeRegexDetector([])
    natasha = FakeNatashaDetector()
    cascade = CascadeDetector(regex_detector=regex, natasha_detector=natasha)
    await cascade.initialize()

    await cascade.detect("Иванов Иван " + "текст " * 2_000)

    assert natasha.deadline is not None
