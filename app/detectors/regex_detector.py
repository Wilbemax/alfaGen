from __future__ import annotations

import bisect
import logging
import re
from collections.abc import Iterable
from typing import ClassVar

from app.config.settings import pii_rules
from app.detectors.base import BaseDetector, DetectorConfig
from app.models.pii import PIIMatch

logger = logging.getLogger(__name__)


class RegexDetector(BaseDetector):
    """Ищет только структурированные типы ПДн, назначенные regex-слою."""

    _CONTEXT_RADIUS = 40
    _DATE = re.compile(
        r"(?<!\d)(?:(?:0?[1-9]|[12]\d|3[01])\.\s*(?:0?[1-9]|1[0-2])\.\s*(?:19|20)\d{2}"
        r"|(?:0?[1-9]|[12]\d|3[01])/(?:0?[1-9]|1[0-2])/(?:19|20)\d{2}"
        r"|(?:0?[1-9]|1[0-2])/(?:0?[1-9]|[12]\d|3[01])/(?:19|20)\d{2}"
        r"|(?:19|20)\d{2}-(?:0?[1-9]|1[0-2])-(?:0?[1-9]|[12]\d|3[01])"
        r"|(?:19|20)\d{2}\.(?:(?:0?[1-9]|1[0-2])\.(?:0?[1-9]|[12]\d|3[01])"
        r"|(?:0?[1-9]|[12]\d|3[01])\.(?:0?[1-9]|1[0-2])))(?!\d)"
    )
    _TEXT_DATE = re.compile(
        r"(?<!\d)(?:0?[1-9]|[12]\d|3[01])\s+"
        r"(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+"
        r"(?:19|20)\d{2}(?:\s+года?\b)?",
        re.IGNORECASE,
    )
    _PASSPORT_VALUE = re.compile(
        r"(?<!\d)(?:\d{4}\s?\d{6}|\d{2}\s\d{2}\s\d{6})(?!\d)"
    )
    _PASSPORT_SPLIT = re.compile(
        r"(?i)\bсер(?:ия|ии|ию)\s+(?P<series>\d{4})\s+номер\s+(?P<number>\d{6})(?!\d)"
    )
    _DRIVER_VALUE = re.compile(
        r"(?<!\d)(?:\d{2}\s\d{2}\s\d{6}|\d{4}\s\d{6}|\d{2}\s\d{6})(?!\d)"
    )
    _CITIZENSHIP_VALUE = re.compile(
        r"(?iu)(?<![а-яёa-z])(?:гражданство|гражданин|гражданка)"
        r"(?![а-яёa-z])(?:[ \t]+(?:заявителя|клиента))?"
        r"[ \t]*(?:[:—-][ \t]*)?"
        r"(?P<value>рф|российск(?:ая|ой|ую)\s+федерац(?:ия|ии|ию)"
        r"|росси(?:я|и|ю)|казахстан(?:а)?|украин(?:а|ы|е|у)"
        r"|(?:республик(?:а|и|у)[ \t]+)?беларус(?:ь|и))"
        r"(?![а-яёa-z])"
    )
    _BIRTH_YEAR_MAX = 2012
    _ISSUER_VALUE = re.compile(
        r"(?iu)(?<![а-яёa-z])(?:о?уфмс|овд|мвд)"
        r"(?:\s+россии)?"
        r"(?:"
        r"\s+по\s+(?:(?:г\.\s*|городу\s+|республике\s+)[а-яё][а-яё-]+|[а-яё][а-яё-]+)"
        r"|\s+(?:района|города)\s+[а-яё][а-яё-]+"
        r")"
    )
    _NAME_WORD = re.compile(r"(?iu)[а-яё]{2,}(?:-[а-яё]{2,})*")
    _PATRONYMIC = re.compile(r"(?iu)(?:ович|евич|овна|евна|ична|инична)$")
    _YEAR = re.compile(r"(?:19|20)\d{2}")
    _CARD_HOLDER_VALUE = re.compile(
        r"(?iu)(?<![а-яёa-z])(?:имя[ \t]+держателя|держатель(?:[ \t]+карты)?|cardholder|holder)"
        r"(?![а-яёa-z])\s*(?:[:—-][ \t]*)?"
        r"(?P<value>[a-z]+[ \t]+[a-z]+)"
        r"(?=$|[\r\n.,;!?]|[ \t]+[а-яё]|[ \t]+(?:pin(?:[ -]?code)?|cvv2?|cvc2?)\b)"
    )

    _TYPE_SPECIFICITY: ClassVar[dict[str, int]] = {
        "BANK_CARD": 120,
        "CARD_CVV": 115,
        "CARD_PIN": 115,
        "PASSPORT": 110,
        "DRIVER_LICENSE": 100,
        "PASSPORT_DEPT_CODE": 95,
        "EMAIL": 85,
        "PHONE": 85,
        "INN": 80,
        "CITIZENSHIP": 80,
        "CARD_HOLDER": 80,
        "PASSPORT_ISSUE_DATE": 75,
        "DATE_OF_BIRTH": 70,
    }

    def __init__(self, config: DetectorConfig | None = None) -> None:
        super().__init__(config)
        self._patterns: dict[str, list[tuple[re.Pattern[str], float]]] = {}
        self._enabled_types: set[str] | None = None

    @property
    def name(self) -> str:
        return "regex"

    @property
    def supported_entity_types(self) -> set[str]:
        return set(self._patterns) | {
            "PASSPORT",
            "DRIVER_LICENSE",
            "DATE_OF_BIRTH",
            "PASSPORT_ISSUE_DATE",
            "CITIZENSHIP",
            "CARD_HOLDER",
            "PERSON",
            "PASSPORT_ISSUER",
        }

    async def initialize(self) -> None:
        default = pii_rules.default or {}
        enabled = default.get("enabled_entity_types", []) or []
        self._enabled_types = set(enabled) if enabled else None
        self._patterns = self._build_patterns()
        self._initialized = True
        logger.info(
            "Regex detector initialized",
            extra={"pattern_count": sum(map(len, self._patterns.values()))},
        )

    @staticmethod
    def _build_patterns() -> dict[str, list[tuple[re.Pattern[str], float]]]:
        return {
            "EMAIL": [
                (
                    re.compile(
                        r"(?<![\w.+-])[A-Za-z0-9][A-Za-z0-9._%+-]{0,63}"
                        r"@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?!\w)"
                    ),
                    0.98,
                ),
            ],
            "PHONE": [
                (
                    re.compile(
                        r"(?<!\d)(?:\+7|7|8)[ -]?\(?\d{3}\)?[ -]?"
                        r"\d{3}[ -]?\d{2}[ -]?\d{2}(?!\d)"
                    ),
                    0.95,
                ),
            ],
            "INN": [
                (
                    re.compile(
                        r"(?iu)(?:\bинн\b|\bналогов(?:ый|ого)\s+номер\b)"
                        r"[^\d]{0,20}(?P<value>\d{10}|\d{12})(?!\d)"
                    ),
                    0.90,
                ),
            ],
            "BANK_CARD": [
                (re.compile(r"(?<!\d)\d{4}(?:[ -]?\d{4}){3}(?!\d)"), 0.95),
            ],
            "CARD_CVV": [
                (
                    re.compile(
                        r"(?i)\b(?:cvv2?|cvc2?)\b[ \t:-]*(?P<value>\d{3})(?!\d)"
                    ),
                    0.95,
                ),
            ],
            "CARD_PIN": [
                (
                    re.compile(
                        r"(?i)\b(?:пин(?:[ -]?код)?|pin(?:[ -]?code)?)\b"
                        r"[ \t:-]*(?P<value>\d{4})(?!\d)"
                    ),
                    0.95,
                ),
            ],
            "PASSPORT_DEPT_CODE": [
                (
                    re.compile(
                        r"(?i)\bкод\s+подразделения\b\s*[:—-]?\s*"
                        r"(?P<value>\d{3}-\d{3})(?!\d)"
                    ),
                    0.95,
                ),
            ],
        }

    async def detect(self, text: str) -> list[PIIMatch]:
        if not self.config.enabled or not text:
            return []

        matches = self._detect_regular_patterns(text)
        matches.extend(self._detect_documents(text))
        matches.extend(self._detect_dates(text))
        matches.extend(self._detect_contextual_values(text))
        matches.extend(self._detect_names(text))
        matches.extend(self._detect_issuers(text))

        enabled = [match for match in matches if self._is_enabled(match.entity_type)]
        deduplicated = list(
            {(match.entity_type, match.start, match.end): match for match in enabled}.values()
        )
        resolved = self._resolve_overlaps(deduplicated)
        return [
            match
            for match in resolved
            if match.confidence >= self.config.confidence_threshold
        ]

    def _detect_regular_patterns(self, text: str) -> list[PIIMatch]:
        found: list[PIIMatch] = []
        for entity_type, patterns in self._patterns.items():
            for pattern, confidence in patterns:
                for match in pattern.finditer(text):
                    start, end = (
                        match.span("value")
                        if "value" in match.groupdict()
                        else match.span()
                    )
                    found.append(
                        self._make_match(entity_type, text, start, end, confidence)
                    )
        return found

    def _detect_documents(self, text: str) -> list[PIIMatch]:
        found: list[PIIMatch] = []
        split_ranges: list[tuple[int, int]] = []

        for match in self._PASSPORT_SPLIT.finditer(text):
            split_ranges.append(match.span())
            for group in ("series", "number"):
                start, end = match.span(group)
                found.append(self._make_match("PASSPORT", text, start, end, 0.98))

        for match in self._PASSPORT_VALUE.finditer(text):
            start, end = match.span()
            if any(
                start < split_end and end > split_start
                for split_start, split_end in split_ranges
            ):
                continue
            if self._has_left_context(text, start, ("паспорт", "серия")):
                found.append(self._make_match("PASSPORT", text, start, end, 0.96))

        for match in self._DRIVER_VALUE.finditer(text):
            start, end = match.span()
            if self._has_left_context(
                text, start, ("водительское", "удостоверение", "права")
            ):
                found.append(
                    self._make_match("DRIVER_LICENSE", text, start, end, 0.93)
                )
        return found

    def _detect_dates(self, text: str) -> list[PIIMatch]:
        found: list[PIIMatch] = []
        for pattern in (self._DATE, self._TEXT_DATE):
            for match in pattern.finditer(text):
                start, end = match.span()
                before = text[max(0, start - self._CONTEXT_RADIUS):start].casefold()
                issue_context = re.search(
                    r"(?:выдано?|дата\s+выдачи)\W*$",
                    before,
                )
                birth_context = re.search(
                    r"(?:дата\s+рождения|год\s+рождения|birth\s+date|родил(?:ся|ась))"
                    r"[^.!?;\r\n]{0,24}$",
                    before,
                    re.IGNORECASE,
                )
                if issue_context:
                    entity_type = "PASSPORT_ISSUE_DATE"
                elif birth_context or self._birth_year(match.group()):
                    entity_type = "DATE_OF_BIRTH"
                else:
                    continue
                found.append(
                    self._make_match(entity_type, text, start, end, 0.92)
                )
        return found

    def _detect_contextual_values(self, text: str) -> list[PIIMatch]:
        found: list[PIIMatch] = []
        contextual_patterns = (
            (
                "CITIZENSHIP",
                self._CITIZENSHIP_VALUE,
                ("гражданство", "гражданин", "гражданка"),
                0.95,
            ),
            (
                "CARD_HOLDER",
                self._CARD_HOLDER_VALUE,
                ("держатель", "держателя", "cardholder", "holder"),
                0.95,
            ),
        )
        for entity_type, pattern, keywords, confidence in contextual_patterns:
            for match in pattern.finditer(text):
                start, end = (
                    match.span("value")
                    if "value" in match.groupdict()
                    else match.span()
                )
                if self._has_context(text, start, end, keywords):
                    found.append(
                        self._make_match(entity_type, text, start, end, confidence)
                    )
        return found

    def _detect_names(self, text: str) -> list[PIIMatch]:
        """Три соседних слова, если одно из них — отчество. Регистр не важен."""
        words = list(self._NAME_WORD.finditer(text))
        found: list[PIIMatch] = []
        index = 0
        while index + 2 < len(words):
            window = words[index:index + 3]
            gaps_are_spaces = all(
                text[window[pos].end():window[pos + 1].start()].isspace()
                for pos in range(2)
            )
            has_patronymic = any(self._PATRONYMIC.search(word.group()) for word in window)
            if gaps_are_spaces and has_patronymic:
                found.append(
                    self._make_match("PERSON", text, window[0].start(), window[2].end(), 0.9)
                )
                index += 3
            else:
                index += 1
        return found

    def _detect_issuers(self, text: str) -> list[PIIMatch]:
        """Орган выдачи по ОУФМС, УФМС, ОВД или МВД, без обязательного «выдан»."""
        return [
            self._make_match("PASSPORT_ISSUER", text, match.start(), match.end(), 0.93)
            for match in self._ISSUER_VALUE.finditer(text)
        ]

    @classmethod
    def _birth_year(cls, value: str) -> bool:
        """Год в диапазоне рождения, если рядом нет слов о выдаче паспорта."""
        matched = cls._YEAR.search(value)
        if matched is None:
            return False
        return int(matched.group()) <= cls._BIRTH_YEAR_MAX

    def _make_match(
        self,
        entity_type: str,
        text: str,
        start: int,
        end: int,
        confidence: float,
    ) -> PIIMatch:
        return PIIMatch(
            entity_type,
            text[start:end],
            start,
            end,
            confidence,
            self.name,
        )

    def _is_enabled(self, entity_type: str) -> bool:
        return self._enabled_types is None or entity_type in self._enabled_types

    def _has_context(
        self,
        text: str,
        start: int,
        end: int,
        keywords: Iterable[str],
    ) -> bool:
        context = text[
            max(0, start - self._CONTEXT_RADIUS):
            min(len(text), end + self._CONTEXT_RADIUS)
        ].casefold()
        return any(
            re.search(
                rf"(?<![а-яёa-z]){re.escape(keyword)}(?![а-яёa-z])",
                context,
            )
            for keyword in keywords
        )

    def _has_left_context(
        self,
        text: str,
        start: int,
        keywords: Iterable[str],
    ) -> bool:
        """Проверяет только предшествующий локальный контекст значения."""
        context = text[max(0, start - self._CONTEXT_RADIUS):start].casefold()
        if re.search(r"\bбез\s+слова\s+паспорт\b", context):
            return False
        return any(
            re.search(
                rf"(?<![а-яёa-z]){re.escape(keyword)}(?![а-яёa-z])",
                context,
            )
            for keyword in keywords
        )

    def _resolve_overlaps(self, matches: list[PIIMatch]) -> list[PIIMatch]:
        """Снимает перекрытия до фильтрации: важнее специфичность и длина."""
        ranked = sorted(
            matches,
            key=lambda item: (
                self._TYPE_SPECIFICITY.get(item.entity_type, 0),
                item.end - item.start,
                item.confidence,
                -item.start,
            ),
            reverse=True,
        )
        accepted: list[PIIMatch] = []
        covered: list[tuple[int, int]] = []
        for match in ranked:
            index = bisect.bisect_left(covered, (match.start, match.end))
            neighbours = covered[max(0, index - 1):index + 1]
            if any(
                match.start < end and match.end > start
                for start, end in neighbours
            ):
                continue
            covered.insert(index, (match.start, match.end))
            accepted.append(match)
        return sorted(accepted, key=lambda item: (item.start, item.end))
