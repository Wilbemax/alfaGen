from __future__ import annotations
import re
import logging
from typing import TYPE_CHECKING

from app.detectors.base import BaseDetector, DetectorConfig
from app.models.pii import PIIMatch

if TYPE_CHECKING:
    from app.config.settings import settings as app_settings

logger = logging.getLogger(__name__)


class RegexDetector(BaseDetector):
    """
    Мощный слой регулярных выражений для детекции ПДн.
    Покрывает все обязательные типы с высокой точностью.
    """

    def __init__(self, config: DetectorConfig | None = None) -> None:
        super().__init__(config)
        self._patterns: dict[str, list[tuple[re.Pattern, float]]] = {}
        self._exclusion_patterns: list[re.Pattern] = []

    @property
    def name(self) -> str:
        return "regex"

    @property
    def supported_entity_types(self) -> set[str]:
        return set(self._patterns.keys())

    async def initialize(self) -> None:
        """Компиляция всех regex паттернов"""
        self._patterns = self._build_patterns()
        self._exclusion_patterns = self._build_exclusions()
        logger.info(f"Regex detector initialized with {sum(len(v) for v in self._patterns.values())} patterns")

    def _build_patterns(self) -> dict[str, list[tuple[re.Pattern, float]]]:
        """Строит словарь паттернов: тип -> [(compiled_regex, confidence)]"""
        patterns: dict[str, list[tuple[re.Pattern, float]]] = {}

        # --- Email ---
        patterns["EMAIL"] = [
            (re.compile(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
                re.IGNORECASE,
            ), 0.98),
        ]

        # --- Телефон (российский формат) ---
        patterns["PHONE"] = [
            (re.compile(
                r"(?<!\d)(?:\+7|8|7)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}(?!\d)",
            ), 0.95),
            (re.compile(
                r"(?<!\d)\+7[\s\-]?\d{3}[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}(?!\d)",
            ), 0.97),
        ]

        # --- ИНН (10 или 12 цифр) ---
        patterns["INN"] = [
            (re.compile(r"(?<!\d)\d{12}(?!\d)"), 0.9),
            (re.compile(r"(?<!\d)\d{10}(?!\d)"), 0.85),
        ]

        # --- Банковская карта (16 цифр, с пробелами или без) ---
        patterns["BANK_CARD"] = [
            (re.compile(
                r"(?<!\d)(?:\d{4}[\s\-]?){3}\d{4}(?!\d)",
            ), 0.95),
            (re.compile(r"(?<!\d)\d{16}(?!\d)"), 0.9),
        ]

        # --- CVV (3 цифры, обычно после "cvv") ---
        patterns["CARD_CVV"] = [
            (re.compile(
                r"(?i)(?:cvv|cvc|cvv2|cvc2)[\s:]*\d{3}\b",
            ), 0.9),
        ]

        # --- Пин-код (4 цифры, обычно после "пин") ---
        patterns["CARD_PIN"] = [
            (re.compile(
                r"(?i)(?:пин[\s-]?код|pin)[\s:]*\d{4}\b",
            ), 0.9),
        ]

        # --- Паспорт РФ: серия (4 цифры) + номер (6 цифр) ---
        patterns["PASSPORT_SERIES"] = [
            (re.compile(r"(?<!\d)\d{4}(?!\d)"), 0.7),
        ]
        patterns["PASSPORT_NUMBER"] = [
            (re.compile(r"(?<!\d)\d{6}(?!\d)"), 0.7),
        ]

        # --- Код подразделения (XXX-XXX) ---
        patterns["PASSPORT_DEPT_CODE"] = [
            (re.compile(r"(?<!\d)\d{3}-\d{3}(?!\d)"), 0.95),
        ]

        # --- Водительское удостоверение (серия 4 + номер 6) ---
        patterns["DRIVER_LICENSE"] = [
            (re.compile(r"(?<!\d)\d{4}\s?\d{6}(?!\d)"), 0.85),
        ]

        # --- Дата рождения (дд.мм.гггг) ---
        patterns["DATE_OF_BIRTH"] = [
            (re.compile(
                r"(?<!\d)(?:0[1-9]|[12]\d|3[01])\.(?:0[1-9]|1[0-2])\.(?:19|20)\d{2}(?!\d)",
            ), 0.9),
            (re.compile(
                r"(?<!\d)(?:0[1-9]|[12]\d|3[01])\.(?:0[1-9]|1[0-2])\.\d{2}(?!\d)",
            ), 0.8),
        ]

        # --- Дата выдачи паспорта (дд.мм.гггг) ---
        patterns["PASSPORT_ISSUE_DATE"] = [
            (re.compile(
                r"(?<!\d)(?:0[1-9]|[12]\d|3[01])\.(?:0[1-9]|1[0-2])\.(?:19|20)\d{2}(?!\d)",
            ), 0.85),
        ]

        # --- Гражданство ---
        patterns["CITIZENSHIP"] = [
            (re.compile(
                r"(?i)\b(?:гражданин|гражданство|гражданка)\s+(?:РФ|Российской\s+Федерации|России)\b",
            ), 0.9),
            (re.compile(r"(?i)\b(?:гражданин|гражданство)\s+[А-ЯЁ][а-яё]+\b"), 0.7),
        ]

        # --- Адрес (частичный) ---
        patterns["ADDRESS_PARTIAL"] = [
            (re.compile(
                r"(?i)\b(?:ул\.?|улица|пр\.?|проспект|пер\.?|переулок|бульвар|наб\.?|набережная|ш\.?|шоссе|пл\.?|площадь)\s+[А-ЯЁ][а-яё\-]+(?:\s+\d+[а-яё]?)?",
            ), 0.85),
            (re.compile(
                r"(?i)\b(?:г\.?|город|пос\.?|поселок|деревня|село|дер\.?)\s+[А-ЯЁ][а-яё\-]+",
            ), 0.8),
        ]

        # --- Адрес (полный) ---
        patterns["ADDRESS"] = [
            (re.compile(
                r"(?i)\b(?:г\.?|город)\s+[А-ЯЁ][а-яё\-]+,\s*(?:ул\.?|улица|пр\.?|проспект)\s+[А-ЯЁ][а-яё\-]+,\s*\d+[а-яё]?(?:\s*,\s*(?:кв\.?|квартира)\s*\d+)?",
            ), 0.9),
        ]

        # --- Имя держателя карты (латиница, 2 слова) ---
        patterns["CARD_HOLDER"] = [
            (re.compile(
                r"(?<!\w)[A-Z]{2,}\s+[A-Z]{2,}(?!\w)",
            ), 0.7),
        ]

        return patterns

    def _build_exclusions(self) -> list[re.Pattern]:
        """Паттерны для исключения ложных срабатываний (адреса банков и т.д.)"""
        return [
            re.compile(r"(?i)(?:офис|отделение|филиал)\s*\d*"),
            re.compile(r"(?i)\b(?:сбербанк|альфа-банк|втб|газпромбанк|т-банк|райффайзен|росбанк|открытие)\b"),
        ]

    async def detect(self, text: str) -> list[PIIMatch]:
        if not self.config.enabled:
            return []

        matches: list[PIIMatch] = []
        seen: set[tuple[str, int, int]] = set()

        for entity_type, pattern_list in self._patterns.items():
            for pattern, confidence in pattern_list:
                for match in pattern.finditer(text):
                    start, end = match.start(), match.end()
                    matched_text = text[start:end]

                    # Проверяем исключения
                    if self._is_excluded(matched_text, start, end, text):
                        continue

                    # Дедупликация (тот же тип, та же позиция)
                    key = (entity_type, start, end)
                    if key in seen:
                        continue
                    seen.add(key)

                    # Контекстное повышение уверенности
                    boosted = self._boost_confidence(entity_type, confidence, start, end, text)

                    matches.append(PIIMatch(
                        entity_type=entity_type,
                        text=matched_text,
                        start=start,
                        end=end,
                        confidence=boosted,
                        detector_name=self.name,
                    ))

        # Возвращаем все совпадения.
        # Разрешение перекрытий выполняется на уровне CompositeDetector,
        # где известны разрешенные типы для конкретной системы.
        return matches

    def _resolve_overlaps(self, matches: list[PIIMatch]) -> list[PIIMatch]:
        """
        Разрешает перекрывающиеся совпадения.
        Приоритет: более специфичный тип > более длинное совпадение > выше уверенность.
        Например, банковская карта (16 цифр) не должна давать ложные PASSPORT_SERIES (4 цифры).
        """
        if len(matches) <= 1:
            return matches

        # Сортируем по специфичности типа DESC, длине DESC, уверенности DESC
        sorted_matches = sorted(
            matches,
            key=lambda m: (
                self._type_specificity.get(m.entity_type, 0),
                m.end - m.start,
                m.confidence,
            ),
            reverse=True,
        )

        result: list[PIIMatch] = []
        covered: list[tuple[int, int]] = []

        for match in sorted_matches:
            # Проверяем перекрытие с уже выбранными
            overlaps = any(
                match.start < c_end and match.end > c_start
                for c_start, c_end in covered
            )
            if overlaps:
                continue
            covered.append((match.start, match.end))
            result.append(match)

        # Возвращаем в порядке появления в тексте
        return sorted(result, key=lambda m: m.start)

    # Специфичность типов: чем выше, тем более специфичный тип.
    # Используется для разрешения перекрытий (специфичный тип выигрывает у общего).
    _type_specificity: dict[str, int] = {
        "BANK_CARD": 100,
        "CARD_CVV": 100,
        "CARD_PIN": 100,
        "DRIVER_LICENSE": 90,
        "PASSPORT_DEPT_CODE": 90,
        "EMAIL": 80,
        "PHONE": 80,
        "INN": 80,
        "DATE_OF_BIRTH": 70,
        "PASSPORT_ISSUE_DATE": 70,
        "ADDRESS": 60,
        "ADDRESS_PARTIAL": 60,
        "CITIZENSHIP": 50,
        "CARD_HOLDER": 50,
        "PASSPORT_NUMBER": 40,
        "PASSPORT_SERIES": 10,
    }

    def _boost_confidence(
        self,
        entity_type: str,
        base_confidence: float,
        start: int,
        end: int,
        full_text: str,
    ) -> float:
        """
        Повышает уверенность на основе контекста вокруг совпадения.
        Например, 4-значное число рядом со словом "паспорт" — это серия паспорта.
        """
        # Контекстные ключевые слова для повышения уверенности
        context_keywords: dict[str, list[str]] = {
            "PASSPORT_SERIES": ["паспорт", "серия", "серию", "серии"],
            "PASSPORT_NUMBER": ["паспорт", "номер", "номеру", "номера"],
            "PASSPORT_ISSUE_DATE": ["выдан", "выдано", "дата выдачи"],
            "DATE_OF_BIRTH": ["родился", "родилась", "дата рождения", "день рождения"],
            "PASSPORT_DEPT_CODE": ["код подразделения", "подразделение"],
            "DRIVER_LICENSE": ["водительское", "удостоверение", "права"],
            "INN": ["инн", "иин"],
            "BANK_CARD": ["карта", "карты", "карту", "банковская"],
            "CARD_CVV": ["cvv", "cvc"],
            "CARD_PIN": ["пин", "pin"],
            "CARD_HOLDER": ["cardholder", "holder", "держатель"],
        }

        keywords = context_keywords.get(entity_type)
        if not keywords:
            return base_confidence

        # Проверяем контекст вокруг совпадения (до 40 символов в обе стороны)
        context_start = max(0, start - 40)
        context_end = min(len(full_text), end + 40)
        context = full_text[context_start:context_end].lower()

        for keyword in keywords:
            if keyword in context:
                # Повышаем уверенность, но не выше 0.98
                return min(base_confidence + 0.2, 0.98)

        return base_confidence

    def _is_excluded(self, matched_text: str, start: int, end: int, full_text: str) -> bool:
        """Проверяет, является ли совпадение исключением (адрес банка и т.д.)"""
        # Проверяем контекст вокруг совпадения
        context_start = max(0, start - 30)
        context_end = min(len(full_text), end + 30)
        context = full_text[context_start:context_end]

        for pattern in self._exclusion_patterns:
            if pattern.search(context):
                return True

        return False