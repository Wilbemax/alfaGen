from __future__ import annotations

import logging
import re

from app.config.settings import PIIRulesConfig, pii_rules
from app.models.pii import PIIMatch

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.5

_TYPE_SPECIFICITY: dict[str, int] = {
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
    "PERSON": 30,
    "PASSPORT_SERIES": 10,
}


class ContextFilter:
    """Снятие перекрытий и применение исключений из pii_rules.yaml."""

    def __init__(self, rules: PIIRulesConfig | None = None) -> None:
        self.rules = rules if rules is not None else pii_rules

    def filter(self, text: str, matches: list[PIIMatch]) -> list[PIIMatch]:
        if not text or not matches:
            return []
        normalized = [
            m
            for m in matches
            if 0 <= m.start < m.end <= len(text)
            and text[m.start:m.end] == m.text
            and 0.0 <= m.confidence <= 1.0
        ]
        resolved = self._resolve_overlaps(normalized)
        resolved = [m for m in resolved if m.confidence >= CONFIDENCE_THRESHOLD]
        filtered = self._apply_exclusions(text, resolved)
        filtered.sort(key=lambda m: m.start)
        return filtered

    def _resolve_overlaps(self, matches: list[PIIMatch]) -> list[PIIMatch]:
        """Более специфичный и более длинный спан побеждает."""
        kept: list[PIIMatch] = []
        ordered = sorted(
            matches,
            key=lambda x: (
                self._specificity(x),
                x.end - x.start,
                x.confidence,
                -x.start,
            ),
            reverse=True,
        )
        for m in ordered:
            if self._is_overlapped(m, kept):
                continue
            kept.append(m)
        return kept

    def _is_overlapped(self, m: PIIMatch, kept: list[PIIMatch]) -> bool:
        for other in kept:
            if m.start >= other.end or other.start >= m.end:
                continue
            return True
        return False

    @staticmethod
    def _specificity(m: PIIMatch) -> int:
        return _TYPE_SPECIFICITY.get(m.entity_type, 0)

    def _apply_exclusions(self, text: str, matches: list[PIIMatch]) -> list[PIIMatch]:
        exclusions = self.rules.default.get("exclusions", {})
        historical = {s.lower() for s in exclusions.get("historical_persons", [])}
        service_words = {s.lower() for s in exclusions.get("service_words", [])}
        bank_branches = {s.lower() for s in exclusions.get("bank_branches", [])}
        bank_patterns = exclusions.get("bank_address_patterns", [])

        result: list[PIIMatch] = []
        for m in matches:
            if m.entity_type == "PERSON":
                person_result = self._person_excluded(text, m, historical, service_words)
                if person_result is True:
                    continue
                if isinstance(person_result, PIIMatch):
                    m = person_result
            elif m.entity_type in ("ADDRESS", "ADDRESS_PARTIAL"):
                if self._address_excluded(text, m, bank_branches, bank_patterns):
                    continue
            elif m.entity_type == "CARD_HOLDER":
                if not self._card_holder_kept(text, m):
                    continue
            result.append(m)
        return result

    def _person_excluded(
        self,
        text: str,
        m: PIIMatch,
        historical: set[str],
        service_words: set[str],
    ) -> bool | PIIMatch:
        window = text[max(0, m.start - 40): min(len(text), m.end + 40)].lower()
        if any(h in window for h in historical):
            return True

        span_text = m.text
        words = span_text.split()
        if not words:
            return True

        # Обрезаем служебные слова с начала и конца.
        start_idx = 0
        end_idx = len(words)
        while start_idx < end_idx and words[start_idx].lower() in service_words:
            start_idx += 1
        while end_idx > start_idx and words[end_idx - 1].lower() in service_words:
            end_idx -= 1

        remaining = words[start_idx:end_idx]
        if len(remaining) < 2:
            return True
        # Если служебное слово осталось внутри — удалить спан.
        if any(w.lower() in service_words for w in remaining):
            return True

        # Служебные слова в начале/конце обрезаем: создаём новый спан.
        if start_idx > 0 or end_idx < len(words):
            pattern = r"\s+".join(re.escape(word) for word in remaining)
            located = re.search(pattern, m.text)
            if located is None:
                return True
            new_start = m.start + located.start()
            new_end = m.start + located.end()
            new_text = m.text[located.start():located.end()]
            if len(new_text.split()) < 2:
                return True
            m = PIIMatch(
                entity_type=m.entity_type,
                text=new_text,
                start=new_start,
                end=new_end,
                confidence=m.confidence,
                detector_name=m.detector_name,
                metadata=m.metadata,
            )
            return m
        return False

    def _address_excluded(
        self,
        text: str,
        m: PIIMatch,
        bank_branches: set[str],
        bank_patterns: list[str],
    ) -> bool:
        window = text[max(0, m.start - 80): min(len(text), m.end + 80)].lower()
        if any(b in window for b in bank_branches):
            return True
        for pattern in bank_patterns:
            try:
                if re.search(pattern, window):
                    return True
            except re.error:
                logger.warning("Invalid bank_address_pattern ignored")
        return False

    def _card_holder_kept(self, text: str, m: PIIMatch) -> bool:
        window = text[max(0, m.start - 40): min(len(text), m.end + 40)].lower()
        return any(
            keyword in window
            for keyword in ("держатель", "держателя", "cardholder", "holder")
        )
