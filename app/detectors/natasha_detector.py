from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Callable
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
    r"(?iu)(?:проживает|индекс|улиц\w*|проспект\w*|город\w*"
    r"|(?<![\w])ул\.?|(?<![\w])д\.|(?<![\w])дом\b|(?<![\w])кв\.?|(?<![\w])г\.)"
)
_BIRTH_CUES = ("место рождения", "родился", "родилась")
_ISSUER_CUES = ("выдан", "выдано", "уфмс", "оуфмс", "мвд", "овд")
_SERVICE_PREFIX = re.compile(r"(?iu)(?:поэт|клиент|уважаемый)\s+")
_ISSUER_PREFIX = re.compile(r"(?iu)(?:выдан|выдано)\s+")
_PLACE_LEFT = re.compile(r"(?iu)(?:городе|город|г\.)\s+$")
_ADDRESS_PART = (
    r"(?:росси[яи]|рф|\d{6}"
    r"|(?:г\.?|город)\s+[а-яё\-]+"
    r"|(?:ул\.?|улица|проспект|пер\.?|переулок|шоссе)\s+[а-яё\-]+"
    r"|(?:д\.?|дом)\s*\d+[а-яё]?"
    r"|(?:кв\.?|квартира)\s*\d+"
    r"|(?:корп\.?|корпус)\s*\d+)"
)
_ADDRESS_CLAUSE = re.compile(
    rf"(?iu){_ADDRESS_PART}(?:\s*,\s*{_ADDRESS_PART})+",
)


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
            raise

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
        matches: list[PIIMatch] = []
        for offset, chunk in _gaps(text, covered):
            if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                break
            if not chunk or not self._has_signal(chunk):
                continue
            for start, end, tag, score in self._ner_runner(chunk):
                absolute_start = offset + start
                absolute_end = offset + end
                if _overlaps(absolute_start, absolute_end, covered):
                    continue
                mapped = self._map_span(text, absolute_start, absolute_end, tag, score)
                if mapped is None or _overlaps(mapped.start, mapped.end, covered):
                    continue
                matches.append(mapped)

        matches = _with_address_clauses(text, matches, covered)
        matches.sort(key=lambda item: (item.start, item.end))
        return matches

    def _load_models(self) -> None:
        from natasha import Doc, Segmenter

        segmenter = Segmenter()
        tagger = _build_tagger()
        self._segmenter = segmenter
        self._tagger = tagger
        self._doc_cls = Doc
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
            window = _window(text, start, end)
            if not any(cue in window for cue in _ISSUER_CUES):
                return None
            start = _consume_prefix(text, start, end, _ISSUER_PREFIX)
            end = _extend_issuer(text, start, end)
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


def _with_address_clauses(
    text: str,
    matches: list[PIIMatch],
    covered: list[tuple[int, int]],
) -> list[PIIMatch]:
    """Склеивает соседние части адреса в один спан."""
    clauses: list[PIIMatch] = []
    for match in _ADDRESS_CLAUSE.finditer(text):
        start, end = match.span()
        if _overlaps(start, end, covered):
            continue
        snippet = text[start:end]
        clauses.append(
            PIIMatch(
                entity_type="ADDRESS",
                text=snippet,
                start=start,
                end=end,
                confidence=0.9,
                detector_name="natasha",
            )
        )
    if not clauses:
        return matches
    kept = [
        item
        for item in matches
        if item.entity_type not in {"ADDRESS", "ADDRESS_PARTIAL"}
        or not any(item.start < clause.end and item.end > clause.start for clause in clauses)
    ]
    return kept + clauses


def _window(text: str, start: int, end: int) -> str:
    return text[max(0, start - 40) : min(len(text), end + 40)].lower()


def _address_cue_for_span(text: str, start: int, end: int) -> bool:
    """Адресный признак берётся у самого спана, а не у всей фразы."""
    left = text[max(0, start - 20) : start]
    span = text[start:end]
    return _ADDRESS_SIGNAL.search(left) is not None or _ADDRESS_SIGNAL.search(span) is not None


def _expand_place_left(text: str, start: int) -> int:
    window = text[max(0, start - 12) : start]
    matched = _PLACE_LEFT.search(window)
    if matched is None:
        return start
    return start - len(matched.group())


def _extend_issuer(text: str, start: int, end: int) -> int:
    """Дочитывает название органа до конца оборота, не обрываясь на «г.»."""
    index = end
    length = len(text)
    while index < length:
        char = text[index]
        if char.isalpha() or char in " \t-«»":
            index += 1
            continue
        if char == "." and _abbreviation_dot(text, start, index):
            index += 1
            continue
        break
    return index


def _abbreviation_dot(text: str, start: int, dot_index: int) -> bool:
    cursor = dot_index - 1
    while cursor >= start and text[cursor].isalpha():
        cursor -= 1
    return 1 <= dot_index - 1 - cursor <= 3


def _consume_prefix(text: str, start: int, end: int, pattern: re.Pattern[str]) -> int:
    while start < end:
        matched = pattern.match(text, start, end)
        if matched is None:
            return start
        start = matched.end()
    return start


def _merge_spans(spans: list[tuple[int, int]], length: int) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        start = max(0, start)
        end = min(length, end)
        if start >= end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _gaps(text: str, covered: list[tuple[int, int]]) -> list[tuple[int, str]]:
    chunks: list[tuple[int, str]] = []
    cursor = 0
    for start, end in covered:
        if cursor < start:
            chunks.append((cursor, text[cursor:start]))
        cursor = max(cursor, end)
    if cursor < len(text):
        chunks.append((cursor, text[cursor:]))
    return chunks


def _overlaps(start: int, end: int, covered: list[tuple[int, int]]) -> bool:
    return any(start < stop and end > begin for begin, stop in covered)
