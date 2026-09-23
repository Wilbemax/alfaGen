from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from app.config.settings import settings
from app.detectors.base import BaseDetector, DetectorConfig
from app.models.pii import PIIMatch

if TYPE_CHECKING:
    from natasha import Doc, Segmenter

logger = logging.getLogger(__name__)

# (start, end, tag, confidence) в координатах переданного фрагмента.
NerRunner = Callable[[str], list[tuple[int, int, str, float]]]

_NAME_SIGNAL = re.compile(r"(?iu)(?<![\w])[а-яё]{2,}(?:\s+[а-яё]{2,})+")
_ADDRESS_SIGNAL = re.compile(
    r"(?iu)(?:проживает|адрес|индекс|улиц\w*|проспект\w*|город\w*"
    r"|корпус\w*|строени\w*|квартир\w*|област\w*|район\w*"
    r"|республик\w*|край\b|шоссе\b|переулок\w*"
    r"|(?<![\w])ул\.?|(?<![\w])пр-т\b|(?<![\w])д\."
    r"|(?<![\w])дом\b|(?<![\w])кв\.?|(?<![\w])корп\."
    r"|(?<![\w])стр\.?|(?<![\w])обл\.?|(?<![\w])р-н\b"
    r"|(?<![\w])пер\.?|(?<![\w])г\.)"
)
_DIRECT_ADDRESS_SPAN_SIGNAL = re.compile(
    r"(?iu)(?:улиц\w*|проспект\w*|шоссе\b|переулок\w*"
    r"|(?<![\w])ул\.?|(?<![\w])пр-т\b|(?<![\w])пер\.?"
    r"|(?<![\w])г\.|город\w*)"
)
_BIRTH_CUES = ("место рождения", "родился", "родилась")
_ISSUER_CUES = ("выдан", "выдано", "кем выдан", "орган", "уфмс", "оуфмс", "мвд", "овд")
_SERVICE_PREFIX = re.compile(
    r"(?iu)(?:поэт|клиент|заявитель|уважаемый|на\s+имя)\s+"
)
_ISSUER_PREFIX = re.compile(
    r"(?iu)(?:(?:кем\s+)?выдано?|орган)\s*:?\s*"
)
_PLACE_LEFT = re.compile(
    r"(?iu)(?:городе|город|г\.|улица|ул\.?|проспект|пр-т|шоссе"
    r"|переулок|пер\.?|область|обл\.?|район|р-н|республика|край)\s+$"
)
_ADDRESS_SERVICE_PREFIX = re.compile(
    r"(?iu)(?:\bпроживает\b|\bадрес(?:\s+регистрации)?\b)[ \t]*:[ \t]*"
)
_STRUCTURED_ADDRESS_COMPONENT = re.compile(
    r"(?iu)^(?:\d{6}"
    r"|(?:росси(?:я|и|ю)|рф|казахстан|беларусь)\b"
    r"|(?:г\.?|город|ул\.?|улица|проспект|пр-т|д\.?|дом|корп\.?|корпус"
    r"|стр\.?|строение|кв\.?|квартира|индекс|обл\.?|область|район|р-н"
    r"|республика|край|шоссе|переулок|пер\.?)(?=[ \t]|$)"
    r"|[а-яё][а-яё .-]*[ \t]+(?:область|район|край|шоссе|переулок)\b)"
)
_ADDRESS_FIELD_BOUNDARY = re.compile(
    r"(?iu)\s+(?=(?:паспорт|код\s+подразделения|инн|телефон|(?:e-?mail|почта)\b"
    r"|карта\b|гражданство|дата\s+рождения|водительское\s+удостоверение|права\b))"
)
_ISSUER_VALUE = re.compile(
    r"(?u)(?i:О?УФМС|УФМС|ОВД|МВД)"
    r"(?:\s+(?i:России))?"
    r"(?:\s+(?i:по)\s+(?:(?i:г)\.\s*|(?i:городу)\s+|(?i:Республике)\s+)?"
    r"[А-ЯЁ][а-яё-]+(?:\s+[А-ЯЁ][а-яё-]+)?"
    r"|\s+(?i:района|города)\s+[А-ЯЁ][а-яё-]+)?"
)
_CONTEXT_PERSON = re.compile(
    r"(?u)(?i:(?<![а-яё])(?:заявитель|клиент|на\s+имя)(?![а-яё]))"
    r"[ \t]+(?P<value>[А-ЯЁ][а-яё-]+(?:[ \t]+[А-ЯЁ][а-яё-]+){2})"
)
_PERSON_WORD = re.compile(r"(?u)[А-ЯЁ][а-яё-]+")


class NatashaDetector(BaseDetector):
    """NER-слой: ФИО, адрес, место рождения и орган выдачи.

    Модель загружается один раз на процесс. Если пакет или словари недоступны,
    детектор выключается и возвращает пустой список. Уже занятые regex-интервалы
    в модель не передаются.
    """

    def __init__(
        self,
        config: DetectorConfig | None = None,
        ner_runner: NerRunner | None = None,
    ) -> None:
        if config is None:
            config = DetectorConfig(
                enabled=settings.natasha_enabled,
                confidence_threshold=settings.natasha_confidence_threshold,
            )
        super().__init__(config)
        self._ner_runner = ner_runner
        self._segmenter: Segmenter | None = None
        self._tagger: object | None = None
        self._doc_cls: type[Doc] | None = None

    @property
    def name(self) -> str:
        return "natasha"

    @property
    def supported_entity_types(self) -> set[str]:
        return {"PERSON", "PLACE_OF_BIRTH", "ADDRESS", "PASSPORT_ISSUER"}

    async def initialize(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        if self._ner_runner is not None:
            return
        try:
            self._load_models()
        except Exception:
            logger.warning("Natasha model load failed, detector disabled")
            self.config.enabled = False
            self._ner_runner = None

    async def detect(
        self,
        text: str,
        occupied: list[tuple[int, int]] | None = None,
        deadline_monotonic: float | None = None,
    ) -> list[PIIMatch]:
        if not text or not self.config.enabled:
            return []
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            return []
        if not self._has_signal(text):
            return []

        if not self._initialized:
            try:
                await self.initialize()
            except Exception:
                return []
        if not self.config.enabled or self._ner_runner is None:
            return []

        covered = _merge_spans(occupied or [], len(text))
        prepared = _mask_occupied(text, covered)
        if not self._has_signal(prepared):
            return []

        matches: list[PIIMatch] = []
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            return []
        try:
            # Natasha NER — CPU-bound и блокирует event loop. Выполняем в
            # отдельном потоке, чтобы не сериализовать все запросы под нагрузкой.
            raw_spans = await asyncio.to_thread(self._ner_runner, prepared)
        except Exception:
            logger.warning("Natasha NER failed, entities omitted")
            return []

        for raw_span in raw_spans:
            if not isinstance(raw_span, (tuple, list)) or len(raw_span) != 4:
                continue
            start, end, tag, score = raw_span
            if not isinstance(start, int) or not isinstance(end, int):
                continue
            if _overlaps(start, end, covered):
                continue
            mapped = self._map_span(text, start, end, str(tag), score)
            if mapped is None or _overlaps(mapped.start, mapped.end, covered):
                continue
            matches.append(mapped)

        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            return []
        expanded = _expand_address_matches(text, matches, covered, self.name)
        expanded.extend(_recover_context_people(text, raw_spans, covered, self.name))
        deduplicated = {
            (match.entity_type, match.start, match.end): match for match in expanded
        }
        return sorted(deduplicated.values(), key=lambda item: (item.start, item.end))

    def _load_models(self) -> None:
        doc_cls, segmenter, tagger = _load_natasha_models()
        self._segmenter = segmenter
        self._tagger = tagger
        self._doc_cls = doc_cls
        self._ner_runner = self._run_natasha

    def _run_natasha(self, text: str) -> list[tuple[int, int, str, float]]:
        if self._doc_cls is None or self._segmenter is None or self._tagger is None:
            return []
        doc = self._doc_cls(text)
        doc.segment(self._segmenter)
        doc.tag_ner(self._tagger)
        found: list[tuple[int, int, str, float]] = []
        for span in doc.spans or []:
            score = getattr(span, "score", None)
            confidence = float(score) if isinstance(score, int | float) else 0.9
            found.append((span.start, span.stop, str(span.type), confidence))
        return found

    def _map_span(
        self,
        text: str,
        start: int,
        end: int,
        tag: str,
        score: float,
    ) -> PIIMatch | None:
        if score < self.config.confidence_threshold:
            return None
        start = max(0, start)
        end = min(len(text), end)
        if start >= end:
            return None

        label = tag.upper()
        if label == "PER":
            start = _consume_prefix(text, start, end, _SERVICE_PREFIX)
            start, end = _expand_person_span(text, start, end)
            entity_type = "PERSON"
        elif label == "LOC":
            start = _expand_place_left(text, start)
            window = _window(text, start, end)
            if any(cue in window for cue in _BIRTH_CUES):
                entity_type = "PLACE_OF_BIRTH"
            elif _address_cue_for_span(text, start, end):
                entity_type = "ADDRESS"
            else:
                return None
        elif label == "ORG":
            if not _issuer_context_for_span(text, start, end):
                return None
            start = _consume_prefix(text, start, end, _ISSUER_PREFIX)
            end = _issuer_end(text, start, end)
            entity_type = "PASSPORT_ISSUER"
        else:
            return None

        if start >= end:
            return None
        snippet = text[start:end]
        if not snippet.strip():
            return None
        return PIIMatch(
            entity_type=entity_type,
            text=snippet,
            start=start,
            end=end,
            confidence=score,
            detector_name=self.name,
        )

    @staticmethod
    def _has_signal(text: str) -> bool:
        return _NAME_SIGNAL.search(text) is not None or _ADDRESS_SIGNAL.search(text) is not None


def _build_tagger() -> object:
    """Собирает NER-теггер из локальных словарей или из моделей пакета Natasha."""
    from natasha import NewsEmbedding, NewsNERTagger

    dict_dir = Path(os.environ.get("NATASHA_DICTS_PATH", ""))
    navec_tar = next(dict_dir.glob("navec*.tar"), None) if dict_dir.is_dir() else None
    ner_tar = next(dict_dir.glob("slovnet_ner*.tar"), None) if dict_dir.is_dir() else None
    if navec_tar is not None and ner_tar is not None:
        return NewsNERTagger(NewsEmbedding(str(navec_tar)), str(ner_tar))
    return NewsNERTagger(NewsEmbedding())


@lru_cache(maxsize=1)
def _load_natasha_models() -> tuple[type[Doc], Segmenter, object]:
    """Создаёт и сохраняет общий комплект моделей Natasha для процесса."""
    from natasha import Doc, Segmenter

    return Doc, Segmenter(), _build_tagger()


def _window(text: str, start: int, end: int) -> str:
    return text[max(0, start - 40) : min(len(text), end + 40)].lower()


def _address_cue_for_span(text: str, start: int, end: int) -> bool:
    """Адресный признак берётся у самого спана, а не у всей фразы."""
    left = text[max(0, start - 20) : start]
    span = text[start:end]
    wider_left = text[max(0, start - 40) : start].casefold()
    has_issuer_context = any(cue in wider_left for cue in _ISSUER_CUES)
    has_strong_address_context = re.search(
        r"(?iu)(?:проживает|адрес|индекс|улиц\w*|проспект\w*|ул\.?|пр-т|шоссе|переулок)",
        left,
    ) is not None
    if has_issuer_context and not has_strong_address_context:
        return False
    return (
        _ADDRESS_SIGNAL.search(left) is not None
        or _DIRECT_ADDRESS_SPAN_SIGNAL.search(span) is not None
    )


def _issuer_context_for_span(text: str, start: int, end: int) -> bool:
    """Требует локальный issuer-маркер перед ORG либо внутри самого ORG."""
    span = text[start:end].casefold()
    if re.search(r"(?u)\b(?:о?уфмс|овд|мвд)\b", span):
        return True
    left = text[max(0, start - 32):start].casefold()
    return re.search(
        r"(?u)(?:(?:кем\s+)?выдано?|орган)\s*[:—-]?\s*$",
        left,
    ) is not None


def _issuer_end(text: str, start: int, ner_end: int) -> int:
    """Расширяет ORG только по грамматике названия органа, не по всему тексту."""
    structured = _ISSUER_VALUE.match(text, start)
    if structured is not None:
        return structured.end()
    return _trim_terminal_punctuation(text, start, ner_end)


def _expand_person_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Дополняет частичный PER соседними компонентами очевидного полного ФИО."""
    left = text[max(0, start - 32):start]
    context = re.search(
        r"(?iu)(?:заявитель|клиент|на\s+имя)\s+$",
        left,
    )
    if context is not None:
        candidate = _CONTEXT_PERSON.search(
            text,
            max(0, start - 32),
            min(len(text), end + 48),
        )
        if (
            candidate is not None
            and candidate.start("value") <= start < candidate.end("value")
        ):
            return candidate.span("value")

    words = list(
        _PERSON_WORD.finditer(
            text,
            max(0, start - 24),
            min(len(text), end + 32),
        )
    )
    containing = [
        index
        for index, word in enumerate(words)
        if word.start() < end and word.end() > start
    ]
    if not containing:
        return start, end
    first = containing[0]
    last = containing[-1]
    while first > 0 and text[words[first - 1].end():words[first].start()].isspace():
        first -= 1
    while (
        last + 1 < len(words)
        and text[words[last].end():words[last + 1].start()].isspace()
    ):
        last += 1
    if last - first + 1 == 3:
        return words[first].start(), words[last].end()
    return start, end


def _recover_context_people(
    text: str,
    raw_spans: list[tuple[int, int, str, float]],
    covered: list[tuple[int, int]],
    detector_name: str,
) -> list[PIIMatch]:
    """Ограниченный fallback для трёхчастного ФИО после сильного контекста."""
    person_anchors = [
        (start, end)
        for start, end, tag, _score in raw_spans
        if str(tag).upper() == "PER"
    ]
    recovered: list[PIIMatch] = []
    for candidate in _CONTEXT_PERSON.finditer(text):
        start, end = candidate.span("value")
        if _overlaps(start, end, covered):
            continue
        # Natasha может не вернуть PER после маскирования соседних полей. Сам
        # fallback всё равно остаётся локальным: ровно три именных компонента
        # после сильного персонального маркера.
        confidence = 0.86 if not person_anchors else 0.92
        recovered.append(
            PIIMatch("PERSON", text[start:end], start, end, confidence, detector_name)
        )
    return recovered


def _expand_place_left(text: str, start: int) -> int:
    window = text[max(0, start - 12) : start]
    matched = _PLACE_LEFT.search(window)
    if matched is None:
        return start
    return start - len(matched.group())


def _expand_address_matches(
    text: str,
    matches: list[PIIMatch],
    covered: list[tuple[int, int]],
    detector_name: str,
) -> list[PIIMatch]:
    expanded: list[PIIMatch] = []
    for match in matches:
        if match.entity_type != "ADDRESS":
            expanded.append(match)
            continue
        start, end = _structured_address_bounds(text, match.start, match.end)
        for safe_start, safe_end in _subtract_occupied(start, end, covered):
            expanded.append(
                PIIMatch(
                    entity_type="ADDRESS",
                    text=text[safe_start:safe_end],
                    start=safe_start,
                    end=safe_end,
                    confidence=match.confidence,
                    detector_name=detector_name,
                )
            )
    return expanded


def _structured_address_bounds(
    text: str,
    address_start: int,
    address_end: int,
) -> tuple[int, int]:
    section_start = max(
        text.rfind(";", 0, address_start),
        text.rfind("\n", 0, address_start),
        text.rfind("\r", 0, address_start),
    ) + 1
    section_end = len(text)
    for separator in (";", "\n", "\r"):
        position = text.find(separator, address_end)
        if position != -1:
            section_end = min(section_end, position)

    components: list[tuple[int, int, str]] = []
    for component in re.finditer(r"[^,]+", text[section_start:section_end]):
        start = section_start + component.start()
        end = section_start + component.end()
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            components.append((start, end, text[start:end]))

    anchor = next(
        (
            index
            for index, (start, end, _) in enumerate(components)
            if address_start < end and address_end > start
        ),
        None,
    )
    if anchor is None:
        return address_start, address_end
    if not _is_address_component(components[anchor]):
        return address_start, address_end

    left = anchor
    while left > 0 and _is_address_component(components[left - 1]):
        left -= 1
    right = anchor
    while right + 1 < len(components) and _is_address_component(components[right + 1]):
        right += 1

    start, _, first_text = components[left]
    prefix = _ADDRESS_SERVICE_PREFIX.search(first_text)
    if prefix is not None:
        start += prefix.end()
    end = components[right][1]
    boundary = _ADDRESS_FIELD_BOUNDARY.search(text, address_end, end)
    if boundary is not None:
        end = boundary.start()
    end = _trim_terminal_punctuation(text, start, end)
    if start >= end:
        return address_start, address_end
    return start, end


def _is_address_component(component: tuple[int, int, str]) -> bool:
    _, _, value = component
    return (
        _ADDRESS_SERVICE_PREFIX.search(value) is not None
        or _STRUCTURED_ADDRESS_COMPONENT.search(value) is not None
    )


def _trim_terminal_punctuation(text: str, start: int, end: int) -> int:
    while end > start and text[end - 1].isspace():
        end -= 1
    while end > start and text[end - 1] in "!?;":
        end -= 1
    if end > start and text[end - 1] == ".":
        token = re.search(r"(?iu)([а-яё]+)\.$", text[start:end])
        if token is None or token.group(1).casefold() not in {
            "г", "ул", "д", "кв", "обл", "корп", "стр", "пер",
        }:
            end -= 1
    return end


def _subtract_occupied(
    start: int,
    end: int,
    covered: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    safe: list[tuple[int, int]] = []
    cursor = start
    for occupied_start, occupied_end in covered:
        if occupied_end <= cursor:
            continue
        if occupied_start >= end:
            break
        if cursor < occupied_start:
            safe.append((cursor, min(occupied_start, end)))
        cursor = max(cursor, occupied_end)
        if cursor >= end:
            break
    if cursor < end:
        safe.append((cursor, end))
    return [(part_start, part_end) for part_start, part_end in safe if part_start < part_end]


def _consume_prefix(text: str, start: int, end: int, pattern: re.Pattern[str]) -> int:
    while start < end:
        matched = pattern.match(text, start, end)
        if matched is None:
            return start
        start = matched.end()
    return start


def _merge_spans(spans: list[tuple[int, int]], length: int) -> list[tuple[int, int]]:
    normalized: list[tuple[int, int]] = []
    for interval in spans:
        if not isinstance(interval, (tuple, list)) or len(interval) != 2:
            continue
        start, end = interval
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        start = max(0, min(length, start))
        end = max(0, min(length, end))
        if start < end:
            normalized.append((start, end))

    merged: list[tuple[int, int]] = []
    for start, end in sorted(normalized):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _mask_occupied(text: str, covered: list[tuple[int, int]]) -> str:
    characters = list(text)
    for start, end in covered:
        characters[start:end] = " " * (end - start)
    return "".join(characters)


def _overlaps(start: int, end: int, covered: list[tuple[int, int]]) -> bool:
    return any(start < stop and end > begin for begin, stop in covered)
