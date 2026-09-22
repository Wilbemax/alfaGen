from __future__ import annotations

import bisect
import logging
import re
from typing import ClassVar

from app.config.settings import pii_rules
from app.detectors.base import BaseDetector, DetectorConfig
from app.models.pii import PIIMatch

logger = logging.getLogger(__name__)


class RegexDetector(BaseDetector):
    """
    Мощный слой регулярных выражений для детекции ПДн.
    Покрывает все обязательные типы с высокой точностью.
    Типы и исключения расширяются через app/config/pii_rules.yaml.
    """

    def __init__(self, config: DetectorConfig | None = None) -> None:
        super().__init__(config)
        self._patterns: dict[str, list[tuple[re.Pattern, float]]] = {}
        self._exclusion_patterns: list[re.Pattern] = []
        self._historical_persons: set[str] = set()
        self._bank_branches: set[str] = set()
        self._enabled_types: set[str] | None = None

    @property
    def name(self) -> str:
        return "regex"

    @property
    def supported_entity_types(self) -> set[str]:
        return set(self._patterns.keys())

    async def initialize(self) -> None:
        """Компиляция всех regex паттернов и загрузка правил из конфигурации."""
        self._load_rules_config()
        self._patterns = self._build_patterns()
        self._exclusion_patterns = self._build_exclusions()
        logger.info(f"Regex detector initialized with {sum(len(v) for v in self._patterns.values())} patterns")

    def _load_rules_config(self) -> None:
        """Загружает типы и исключения из app/config/pii_rules.yaml."""
        default = pii_rules.default or {}
        exclusions = default.get("exclusions", {}) or {}

        historical = exclusions.get("historical_persons", []) or []
        self._historical_persons = {h.lower() for h in historical} or self._default_historical_persons

        bank_branches = exclusions.get("bank_branches", []) or []
        self._bank_branches = {b.lower() for b in bank_branches}

        enabled = default.get("enabled_entity_types", []) or []
        self._enabled_types = set(enabled) if enabled else None

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
                r"(?i)(?:cvv|cvc|cvv2|cvc2)[\s:]*(?P<value>\d{3})\b",
            ), 0.9),
        ]

        # --- Пин-код (4 цифры, обычно после "пин") ---
        patterns["CARD_PIN"] = [
            (re.compile(
                r"(?i)(?:пин[\s-]?код|пин|pin)[\s:]*(?P<value>\d{4})\b",
            ), 0.9),
        ]

        # --- Паспорт РФ: серия (4 цифры) + номер (6 цифр) ---
        # Базовая уверенность низкая: 4-значное число само по себе — это год/количество,
        # а не серия паспорта. Уверенность повышается контекстом («серия», «паспорт»).
        patterns["PASSPORT_SERIES"] = [
            (re.compile(r"(?<!\d)\d{4}(?!\d)"), 0.4),
        ]
        patterns["PASSPORT_NUMBER"] = [
            (re.compile(r"(?<!\d)\d{6}(?!\d)"), 0.4),
        ]

        # --- Паспорт целиком: серия + номер одним спаном (когда цифры стоят подряд) ---
        patterns["PASSPORT"] = [
            (re.compile(
                r"(?i)(?:паспорт\s+|серия\s+)(?P<value>\d{4}\s?\d{6})(?!\d)",
            ), 0.9),
        ]

        # --- Код подразделения (XXX-XXX) ---
        patterns["PASSPORT_DEPT_CODE"] = [
            (re.compile(r"(?<!\d)\d{3}-\d{3}(?!\d)"), 0.95),
        ]

        # --- Водительское удостоверение (серия 4 или 2 + номер 6) ---
        patterns["DRIVER_LICENSE"] = [
            (re.compile(r"(?<!\d)\d{4}\s?\d{6}(?!\d)"), 0.85),
            (re.compile(r"(?<!\d)\d{2}\s?\d{6}(?!\d)"), 0.8),
        ]

        # --- Дата рождения (дд.мм.гггг, гггг.дд.мм, словами) ---
        patterns["DATE_OF_BIRTH"] = [
            (re.compile(
                r"(?<!\d)(?:0[1-9]|[12]\d|3[01])\.(?:0[1-9]|1[0-2])\.(?:19|20)\d{2}(?!\d)",
            ), 0.9),
            (re.compile(
                r"(?<!\d)(?:0[1-9]|[12]\d|3[01])\.(?:0[1-9]|1[0-2])\.\d{2}(?!\d)",
            ), 0.8),
            (re.compile(
                r"(?<!\d)(?:19|20)\d{2}\.(?:0[1-9]|1[0-2])\.(?:0[1-9]|[12]\d|3[01])(?!\d)",
            ), 0.85),
            (re.compile(
                r"(?i)(?<!\d)(?:0[1-9]|[12]\d|3[01])\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+(?:19|20)\d{2}\s+года?(?!\w)",
            ), 0.9),
        ]

        # --- Дата выдачи паспорта (дд.мм.гггг, гггг.дд.мм, словами) ---
        patterns["PASSPORT_ISSUE_DATE"] = [
            (re.compile(
                r"(?<!\d)(?:0[1-9]|[12]\d|3[01])\.(?:0[1-9]|1[0-2])\.(?:19|20)\d{2}(?!\d)",
            ), 0.85),
            (re.compile(
                r"(?<!\d)(?:19|20)\d{2}\.(?:0[1-9]|1[0-2])\.(?:0[1-9]|[12]\d|3[01])(?!\d)",
            ), 0.8),
            (re.compile(
                r"(?i)(?<!\d)(?:0[1-9]|[12]\d|3[01])\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+(?:19|20)\d{2}\s+года?(?!\w)",
            ), 0.85),
        ]

        # --- Гражданство ---
        patterns["CITIZENSHIP"] = [
            (re.compile(
                r"(?i)\b(?:гражданин|гражданство|гражданка)\s*[:]?\s*(?P<value>(?:РФ|Российской\s+Федерации|Российская\s+Федерация|России))\b",
            ), 0.9),
            (re.compile(r"(?i)\b(?:гражданин|гражданство)\s*[:]?\s*(?P<value>[А-ЯЁ][а-яё]+)\b"), 0.7),
        ]

        # --- Место рождения ---
        patterns["PLACE_OF_BIRTH"] = [
            (re.compile(
                r"(?i)\b(?:место\s+рождения|родился|родилась|родился\s+в|родилась\s+в)\s*[:]?\s*(?P<value>(?:г\.?|город|пос\.?|поселок|деревня|село|дер\.?)?\s*[А-ЯЁ][а-яё\-]+)",
            ), 0.85),
        ]

        # --- Орган, выдавший паспорт ---
        patterns["PASSPORT_ISSUER"] = [
            (re.compile(
                r"\b(?i:выдан|выдано|орган\s+выдавший|кем\s+выдан)\s*(?i:паспорт\s*[:]?\s*)?(?P<value>[А-ЯЁ][А-ЯЁа-яё\s\-]{3,60})",
            ), 0.8),
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

        # --- ФИО (2-3 слова с заглавной буквы, регистр не важен) ---
        patterns["PERSON"] = [
            (re.compile(
                r"(?<![А-ЯЁа-яё])(?!(?:Клиент|Уважаемый|Гражданин|Гражданка|Господин|Госпожа|Товарищ|Дорогой|Дорогая|Уважаемая)\b)"
                r"(?:[А-ЯЁа-яё]+(?:\s+[А-ЯЁа-яё]+){1,2})(?![а-яё])",
                re.IGNORECASE,
            ), 0.75),
        ]

        return patterns

    def _build_exclusions(self) -> list[re.Pattern]:
        """Паттерны для исключения ложных срабатываний (адреса банков и т.д.)"""
        patterns: list[re.Pattern] = [
            re.compile(r"(?i)(?:офис|отделение|филиал)\s*\d*"),
        ]

        # Названия банков из конфигурации (без хвостового \b, чтобы ловить склонения:
        # «Альфа-Банка», «Альфа-Банке» и т.д.)
        if self._bank_branches:
            branches = "|".join(re.escape(b) for b in sorted(self._bank_branches, key=len, reverse=True))
            patterns.append(re.compile(rf"(?i)\b(?:{branches})"))

        # Паттерны адресов банков из конфигурации
        default = pii_rules.default or {}
        exclusions = default.get("exclusions", {}) or {}
        bank_address_patterns = exclusions.get("bank_address_patterns", []) or []
        for pat in bank_address_patterns:
            try:
                patterns.append(re.compile(pat))
            except re.error:
                logger.warning(f"Invalid bank_address_pattern: {pat}")

        return patterns

    async def detect(self, text: str) -> list[PIIMatch]:
        if not self.config.enabled:
            return []

        matches: list[PIIMatch] = []
        seen: set[tuple[str, int, int]] = set()

        for entity_type, pattern_list in self._patterns.items():
            # Фильтруем по включённым типам из конфигурации
            if self._enabled_types is not None and entity_type not in self._enabled_types:
                continue
            for pattern, confidence in pattern_list:
                for match in pattern.finditer(text):
                    # Если паттерн имеет именованные группы "value*", спан = только значение ПДн
                    # (слова-метки «паспорт», «CVV», «пин-код» и т.д. остаются в тексте).
                    # Несколько групп (value, value2, ...) объединяются в один составной спан.
                    groupdict = match.groupdict()
                    if "value" in groupdict:
                        value_parts: list[str] = []
                        value_start: int | None = None
                        value_end: int | None = None
                        for gname, gval in groupdict.items():
                            if gname.startswith("value") and gval is not None:
                                gs, ge = match.span(gname)
                                if value_start is None or gs < value_start:
                                    value_start = gs
                                if value_end is None or ge > value_end:
                                    value_end = ge
                                value_parts.append(gval)
                        start, end = value_start, value_end
                        matched_text = " ".join(value_parts)
                    else:
                        start, end = match.start(), match.end()
                        matched_text = text[start:end]

                    # Проверяем исключения
                    if self._is_excluded(entity_type, matched_text, start, end, text):
                        continue

                    # Исключаем исторических личностей (не ПДн)
                    if entity_type == "PERSON" and self._is_historical_person(matched_text):
                        continue

                    # Исключаем фразы из общих слов (не ФИО)
                    if entity_type == "PERSON" and (
                        self._is_stopword_phrase(matched_text) or not self._looks_like_name(matched_text)
                    ):
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

        # Сначала снимаем перекрытия: побеждает более специфичный и более длинный спан.
        matches = self._resolve_overlaps(matches)

        # Порог уверенности применяем после снятия перекрытий.
        return [m for m in matches if m.confidence >= self.config.confidence_threshold]

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
        # Покрытые интервалы, отсортированные по start (для O(log n) проверки перекрытия)
        covered: list[tuple[int, int]] = []

        for match in sorted_matches:
            if self._overlaps_covered(match.start, match.end, covered):
                continue
            # Вставляем интервал в отсортированную позицию
            idx = bisect.bisect_left(covered, (match.start, match.end))
            covered.insert(idx, (match.start, match.end))
            result.append(match)

        # Возвращаем в порядке появления в тексте
        return sorted(result, key=lambda m: m.start)

    def _overlaps_covered(self, start: int, end: int, covered: list[tuple[int, int]]) -> bool:
        """Проверяет перекрытие с уже покрытыми интервалами за O(log n)."""
        if not covered:
            return False
        # Находим первый интервал с start >= нашего start
        idx = bisect.bisect_right(covered, (start, end))
        # Проверяем соседние интервалы слева и справа
        for i in (idx - 1, idx):
            if 0 <= i < len(covered):
                c_start, c_end = covered[i]
                if start < c_end and end > c_start:
                    return True
        return False

    # Специфичность типов: чем выше, тем более специфичный тип.
    # Используется для разрешения перекрытий (специфичный тип выигрывает у общего).
    _type_specificity: ClassVar[dict[str, int]] = {
        "BANK_CARD": 100,
        "CARD_CVV": 100,
        "CARD_PIN": 100,
        "PASSPORT": 95,
        "DRIVER_LICENSE": 90,
        "PASSPORT_DEPT_CODE": 90,
        "PASSPORT_ISSUER": 85,
        "EMAIL": 80,
        "PHONE": 80,
        "INN": 80,
        "DATE_OF_BIRTH": 70,
        "PASSPORT_ISSUE_DATE": 70,
        "PLACE_OF_BIRTH": 65,
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
            "PASSPORT": ["паспорт", "серия", "серию", "серии", "номер"],
            "PASSPORT_SERIES": ["паспорт", "серия", "серию", "серии"],
            "PASSPORT_NUMBER": ["паспорт", "номер", "номеру", "номера"],
            "PASSPORT_ISSUER": ["выдан", "выдано", "орган", "кем выдан"],
            "PASSPORT_ISSUE_DATE": ["выдан", "выдано", "дата выдачи"],
            "DATE_OF_BIRTH": ["родился", "родилась", "дата рождения", "день рождения"],
            "PLACE_OF_BIRTH": ["место рождения", "родился", "родилась"],
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

    def _is_excluded(self, entity_type: str, matched_text: str, start: int, end: int, full_text: str) -> bool:
        """Проверяет, является ли совпадение исключением (адрес банка и т.д.)"""
        # Исключение адресов банков применяется только к адресным типам,
        # чтобы «банк» в «банковская карта» не выбивал email/телефон/ИНН.
        if entity_type in ("ADDRESS", "ADDRESS_PARTIAL"):
            # Проверяем контекст вокруг совпадения (широкое окно, чтобы поймать
            # «Офис Сбербанка находится по адресу: ...»)
            context_start = max(0, start - 80)
            context_end = min(len(full_text), end + 80)
            context = full_text[context_start:context_end]

            for pattern in self._exclusion_patterns:
                if pattern.search(context):
                    return True

            # Если само совпадение содержит название банка — исключаем
            if self._bank_branches:
                lowered = matched_text.lower()
                if any(b in lowered for b in self._bank_branches):
                    return True

        return False

    def _is_historical_person(self, matched_text: str) -> bool:
        """Проверяет, является ли ФИО исторической личностью (не ПДн)."""
        words = matched_text.split()
        return any(w.lower() in self._historical_persons for w in words)

    def _is_stopword_phrase(self, matched_text: str) -> bool:
        """Проверяет, состоит ли фраза целиком из общих слов (не ФИО)."""
        words = [w.lower() for w in matched_text.split()]
        if not words:
            return True
        return all(w in self._person_stopwords for w in words)

    def _looks_like_name(self, matched_text: str) -> bool:
        """Проверяет, что хотя бы одно слово похоже на русское имя/фамилию."""
        words = [w.lower() for w in matched_text.split()]
        return any(
            w.endswith(self._name_suffixes) for w in words
        )

    # Типичные окончания русских фамилий, имён и отчеств
    _name_suffixes: tuple[str, ...] = (
        "ов", "ев", "ёв", "ин", "ын", "ский", "цкий", "ской", "ой",
        "ова", "ева", "ина", "ская", "цкая", "ая",
        "ич", "вна", "чна", "ия", "ья", "ий", "ей",
    )

    # Общие слова, которые не являются именами (для отсечения ложных ФИО)
    _person_stopwords: ClassVar[set[str]] = {
        "дата", "рождения", "родился", "родилась", "в", "году", "было",
        "место", "выдан", "выдано", "выдавший", "орган", "кем", "гражданин", "гражданство",
        "гражданка", "адрес", "проживает", "проживающий", "проживающая", "зарегистрирован",
        "зарегистрирована", "паспорт", "серия", "номер", "код", "подразделения",
        "телефон", "email", "инн", "карта", "карты", "карту", "банковская",
        "уважаемый", "уважаемая", "клиент", "господин", "госпожа", "товарищ",
        "дорогой", "дорогая", "это", "этот", "эта", "что", "как", "для", "при",
        "по", "на", "из", "от", "до", "с", "со", "без", "над", "под", "о", "об",
        "и", "или", "а", "но", "не", "ни", "же", "бы", "ли", "то", "все", "всё",
        "его", "её", "их", "мой", "твой", "наш", "ваш", "свой", "который", "которая",
        "которые", "быть", "есть", "стал", "стала", "стало", "был", "была", "были",
        # Организации и аббревиатуры
        "овд", "уфмс", "гувд", "мвд", "фмс", "увд", "отдел", "отделение", "управление",
        "района", "район", "города", "город", "области", "область", "края", "республики",
        # Страны и государства
        "российская", "федерация", "россии", "россия", "рф", "ссср", "советский", "союз",
    }

    # Исторические личности, чьи ФИО не считаются персональными данными.
    # Используется как fallback, если список не задан в pii_rules.yaml.
    _default_historical_persons: ClassVar[set[str]] = {
        "пушкин", "лермонтов", "толстой", "достоевский", "чехов", "гоголь",
        "тургенев", "бунин", "шолохов", "пастернак", "солженицын", "бродский",
        "ахматова", "цветаева", "маяковский", "есенин", "блок", "горький",
        "ленин", "сталин", "хрущев", "брежнев", "горбачев", "ельцин",
        "путин", "медведев", "навальный", "шойгу", "лавров", "мишустин",
    }
