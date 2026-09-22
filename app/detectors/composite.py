from __future__ import annotations
import logging
from typing import TYPE_CHECKING

from app.detectors.base import BaseDetector, DetectorConfig
from app.detectors.natasha_detector import NatashaDetector
from app.detectors.presidio_detector import PresidioDetector
from app.detectors.regex_detector import RegexDetector
from app.models.pii import PIIMatch

if TYPE_CHECKING:
    from app.config.settings import settings as app_settings

logger = logging.getLogger(__name__)


class CompositeDetector(BaseDetector):
    """
    Композитный детектор: объединяет Natasha + Presidio + Regex.
    Стратегии:
      - priority: детекторы запускаются по приоритету, первый нашедший сущность выигрывает
      - union: все детекторы, дедупликация по позиции
      - intersection: только сущности, найденные всеми детекторами
    """

    def __init__(self, config: DetectorConfig | None = None) -> None:
        super().__init__(config)
        self._detectors: list[BaseDetector] = []
        self._strategy = "priority"
        self._deduplicate = True
        self._min_confidence = 0.75

    @property
    def name(self) -> str:
        return "composite"

    @property
    def supported_entity_types(self) -> set[str]:
        types: set[str] = set()
        for detector in self._detectors:
            types.update(detector.supported_entity_types)
        return types

    def configure(
        self,
        strategy: str = "priority",
        deduplicate: bool = True,
        min_confidence: float = 0.75,
        natasha_enabled: bool = True,
        presidio_enabled: bool = True,
        regex_enabled: bool = True,
    ) -> None:
        """Настройка композитного детектора"""
        self._strategy = strategy
        self._deduplicate = deduplicate
        self._min_confidence = min_confidence

        self._detectors = []
        if natasha_enabled:
            self._detectors.append(NatashaDetector(DetectorConfig(
                enabled=True,
                confidence_threshold=0.85,
                priority=1,
            )))
        if presidio_enabled:
            self._detectors.append(PresidioDetector(DetectorConfig(
                enabled=True,
                confidence_threshold=0.8,
                priority=2,
            )))
        if regex_enabled:
            self._detectors.append(RegexDetector(DetectorConfig(
                enabled=True,
                confidence_threshold=0.75,
                priority=3,
            )))

        # Сортируем по приоритету
        self._detectors.sort(key=lambda d: d.config.priority)

    async def initialize(self) -> None:
        """Инициализация всех поддетекторов"""
        for detector in self._detectors:
            await detector.initialize()
        logger.info(f"Composite detector initialized with {len(self._detectors)} sub-detectors")

    async def detect(self, text: str) -> list[PIIMatch]:
        if not self._detectors:
            return []

        if self._strategy == "priority":
            return await self._detect_priority(text)
        elif self._strategy == "union":
            return await self._detect_union(text)
        elif self._strategy == "intersection":
            return await self._detect_intersection(text)
        else:
            return await self._detect_priority(text)

    async def _detect_priority(self, text: str) -> list[PIIMatch]:
        """Детекторы по приоритету: первый нашедший сущность выигрывает"""
        all_matches: list[PIIMatch] = []
        covered: set[tuple[int, int]] = set()

        for detector in self._detectors:
            matches = await detector(text)
            for match in matches:
                # Проверяем, не покрыта ли позиция уже найденной сущностью
                if self._is_covered(match.start, match.end, covered):
                    continue
                covered.add((match.start, match.end))
                all_matches.append(match)

        return self._filter_by_confidence(all_matches)

    async def _detect_union(self, text: str) -> list[PIIMatch]:
        """Все детекторы, дедупликация по позиции"""
        all_matches: list[PIIMatch] = []
        for detector in self._detectors:
            matches = await detector(text)
            all_matches.extend(matches)

        if self._deduplicate:
            all_matches = self._deduplicate_matches(all_matches)

        return self._filter_by_confidence(all_matches)

    async def _detect_intersection(self, text: str) -> list[PIIMatch]:
        """Только сущности, найденные всеми детекторами"""
        if not self._detectors:
            return []

        # Запускаем все детекторы
        all_results: list[list[PIIMatch]] = []
        for detector in self._detectors:
            matches = await detector(text)
            all_results.append(matches)

        # Находим пересечение по позициям
        if len(all_results) < 2:
            return self._filter_by_confidence(all_results[0] if all_results else [])

        # Строим карту позиций для первого детектора
        first = all_results[0]
        position_map: dict[tuple[int, int], PIIMatch] = {}
        for match in first:
            position_map[(match.start, match.end)] = match

        # Проверяем, что позиция есть во всех остальных
        result: list[PIIMatch] = []
        for pos, match in position_map.items():
            found_in_all = True
            for other in all_results[1:]:
                if not any(m.start == pos[0] and m.end == pos[1] for m in other):
                    found_in_all = False
                    break
            if found_in_all:
                result.append(match)

        return self._filter_by_confidence(result)

    def _is_covered(self, start: int, end: int, covered: set[tuple[int, int]]) -> bool:
        """Проверяет, перекрывается ли позиция с уже покрытыми"""
        for c_start, c_end in covered:
            if start < c_end and end > c_start:
                return True
        return False

    def _deduplicate_matches(self, matches: list[PIIMatch]) -> list[PIIMatch]:
        """Дедупликация: оставляем сущность с максимальной уверенностью для каждой позиции"""
        best_by_pos: dict[tuple[int, int], PIIMatch] = {}
        for match in matches:
            pos = (match.start, match.end)
            if pos not in best_by_pos or match.confidence > best_by_pos[pos].confidence:
                best_by_pos[pos] = match
        return list(best_by_pos.values())

    def _filter_by_confidence(self, matches: list[PIIMatch]) -> list[PIIMatch]:
        return [m for m in matches if m.confidence >= self._min_confidence]

    def filter_by_types(self, matches: list[PIIMatch], allowed_types: set[str]) -> list[PIIMatch]:
        """Фильтрация по разрешенным типам"""
        return [m for m in matches if m.entity_type in allowed_types]

    def resolve_overlaps(self, matches: list[PIIMatch]) -> list[PIIMatch]:
        """
        Разрешает перекрывающиеся совпадения ПОСЛЕ фильтрации по типам.
        Приоритет: более специфичный тип > более длинное совпадение > выше уверенность.
        Это должно выполняться после filter_by_types, чтобы не терять
        менее специфичные типы (например, PASSPORT_SERIES), когда более
        специфичный тип (DRIVER_LICENSE) отфильтрован конфигурацией системы.
        """
        if len(matches) <= 1:
            return matches

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
            overlaps = any(
                match.start < c_end and match.end > c_start
                for c_start, c_end in covered
            )
            if overlaps:
                continue
            covered.append((match.start, match.end))
            result.append(match)

        return sorted(result, key=lambda m: m.start)

    # Специфичность типов для разрешения перекрытий
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

    def apply_exclusions(self, matches: list[PIIMatch], exclusions: dict) -> list[PIIMatch]:
        """Применяет исключения из конфигурации (исторические личности, адреса банков)"""
        if not exclusions:
            return matches

        historical = set(exclusions.get("historical_persons", []))
        bank_branches = set(exclusions.get("bank_branches", []))

        result: list[PIIMatch] = []
        for match in matches:
            # Проверяем исторические личности
            if match.entity_type == "PERSON":
                words = match.text.split()
                if any(w in historical for w in words):
                    continue

            # Проверяем адреса банков
            if match.entity_type in ("ADDRESS", "ADDRESS_PARTIAL"):
                if any(b.lower() in match.text.lower() for b in bank_branches):
                    continue

            result.append(match)

        return result