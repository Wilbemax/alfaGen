from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import ClassVar

from app.models.pii import PIIMatch


@dataclass(slots=True)
class MaskResult:
    """Результат маскирования."""
    masked_text: str
    # Спаны в замаскированном тексте: (mask_start, mask_end, original_span)
    spans: list[tuple[int, int, str]] = field(default_factory=list)
    # Типы сущностей в порядке появления в тексте
    entity_types: list[str] = field(default_factory=list)


class Masker:
    """
    Маскирование и демаскирование текста.

    Маскирование заменяет каждый спам ПДн на частичную маску (инициалы, звёздочки),
    сохраняя неперсональный текст посимвольно. Для обратимости возвращает спаны
    (позиции масок в замаскированном тексте + оригиналы), которые хранятся по payload_id.

    Демаскирование восстанавливает оригиналы по сохранённым спанам.
    """

    def mask(self, text: str, entities: list[PIIMatch]) -> MaskResult:
        """Маскирует сущности в тексте, возвращая маску и спаны для восстановления."""
        if not entities:
            return MaskResult(masked_text=text)

        # Сортируем по start позиции ASC (слева направо)
        sorted_entities = sorted(entities, key=lambda e: e.start)

        result_parts: list[str] = []
        spans: list[tuple[int, int, str]] = []
        entity_types: list[str] = []
        orig_pos = 0
        masked_pos = 0

        for entity in sorted_entities:
            # Копируем неизменённый текст до текущей сущности
            unchanged = text[orig_pos : entity.start]
            result_parts.append(unchanged)
            masked_pos += len(unchanged)

            # Вставляем маску
            mask = self._mask_span(entity.entity_type, entity.text)
            result_parts.append(mask)
            spans.append((masked_pos, masked_pos + len(mask), entity.text))
            entity_types.append(entity.entity_type)
            masked_pos += len(mask)

            orig_pos = entity.end

        # Копируем оставшийся текст
        result_parts.append(text[orig_pos:])

        return MaskResult(masked_text="".join(result_parts), spans=spans, entity_types=entity_types)

    def unmask(self, masked_text: str, spans: list[tuple[int, int, str]]) -> str:
        """
        Демаскирует текст по сохранённым спанам.
        Заменяет каждый спам маски на оригинал (справа налево, чтобы не сбить индексы).
        """
        if not spans:
            return masked_text

        result = masked_text
        # Сортируем по позиции DESC
        for mask_start, mask_end, original in sorted(spans, key=lambda s: s[0], reverse=True):
            # Если текст изменился (например, ретрай с другим содержимым) — пропускаем
            if mask_end > len(result):
                continue
            result = result[:mask_start] + original + result[mask_end:]

        return result

    def _mask_span(self, entity_type: str, original: str) -> str:
        """Возвращает частичную маску для спана ПДн."""
        handler_name = self._maskers.get(entity_type)
        if handler_name is not None:
            handler = getattr(self, handler_name)
            return handler(original)
        # По умолчанию — полностью скрываем
        return self._mask_all(original)

    def _mask_all(self, original: str) -> str:
        """Полностью скрывает спам, сохраняя длину."""
        return "*" * len(original)

    def _mask_initials(self, original: str) -> str:
        """ФИО → инициалы: «Иванов Иван Иванович» → «И. И. И.»"""
        words = [w for w in re.split(r"\s+", original.strip()) if w]
        if not words:
            return self._mask_all(original)
        initials = " ".join(f"{w[0].upper()}." for w in words)
        return initials

    def _mask_keep_edges(self, original: str, keep_start: int = 2, keep_end: int = 2) -> str:
        """Скрывает середину, сохраняя первые keep_start и последние keep_end символов."""
        length = len(original)
        if length <= keep_start + keep_end:
            return self._mask_all(original)
        return original[:keep_start] + "*" * (length - keep_start - keep_end) + original[length - keep_end :]

    def _mask_passport(self, original: str) -> str:
        """Паспорт → сохраняем первые 2 и последние 2 цифры: «4509 123456» → «45******56»"""
        digits = re.findall(r"\d", original)
        if not digits:
            return self._mask_all(original)
        keep_start = min(2, len(digits))
        keep_end = min(2, len(digits) - keep_start)
        result = []
        for i, d in enumerate(digits):
            if i < keep_start or i >= len(digits) - keep_end:
                result.append(d)
            else:
                result.append("*")
        return "".join(result)

    def _mask_email(self, original: str) -> str:
        """Email → сохраняем первую букву и домен: «ivan.ivanov@example.com» → «i***@example.com»"""
        at = original.find("@")
        if at <= 0:
            return self._mask_all(original)
        local = original[:at]
        domain = original[at:]
        if len(local) <= 1:
            return self._mask_all(original)
        return local[0] + "*" * (len(local) - 1) + domain

    def _mask_phone(self, original: str) -> str:
        """Телефон → сохраняем код страны и последние 2 цифры."""
        digits = re.findall(r"\d", original)
        if not digits:
            return self._mask_all(original)
        # Сохраняем первые 2 цифры (код страны) и последние 2
        keep_start = min(2, len(digits))
        keep_end = min(2, len(digits) - keep_start)
        # Строим маску посимвольно, сохраняя нецифровые разделители
        result = []
        digit_idx = 0
        total = len(digits)
        for ch in original:
            if ch.isdigit():
                if digit_idx < keep_start or digit_idx >= total - keep_end:
                    result.append(ch)
                else:
                    result.append("*")
                digit_idx += 1
            else:
                result.append(ch)
        return "".join(result)

    def _mask_card(self, original: str) -> str:
        """Банковская карта → сохраняем первые 4 и последние 4 цифры."""
        digits = re.findall(r"\d", original)
        if not digits:
            return self._mask_all(original)
        keep_start = min(4, len(digits))
        keep_end = min(4, len(digits) - keep_start)
        result = []
        digit_idx = 0
        total = len(digits)
        for ch in original:
            if ch.isdigit():
                if digit_idx < keep_start or digit_idx >= total - keep_end:
                    result.append(ch)
                else:
                    result.append("*")
                digit_idx += 1
            else:
                result.append(ch)
        return "".join(result)

    def _mask_date(self, original: str) -> str:
        """Дата → скрываем день и месяц, сохраняем год: «12.05.1990» → «**.**.1990»"""
        parts = re.split(r"(\D+)", original)
        result = []
        for part in parts:
            if part.isdigit() and len(part) == 4:
                result.append(part)
            elif part.isdigit():
                result.append("*" * len(part))
            else:
                result.append(part)
        return "".join(result)

    def _mask_dept_code(self, original: str) -> str:
        """Код подразделения → «770-123» → «***-***»"""
        return re.sub(r"\d", "*", original)

    def _mask_cvv(self, original: str) -> str:
        """CVV → «***»"""
        return re.sub(r"\d", "*", original)

    def _mask_pin(self, original: str) -> str:
        """Пин-код → «****»"""
        return re.sub(r"\d", "*", original)

    # Маппинг типов ПДн на методы маскирования
    _maskers: ClassVar[dict[str, str]] = {
        "PERSON": "_mask_initials",
        "CARD_HOLDER": "_mask_initials",
        "PASSPORT": "_mask_passport",
        "PASSPORT_SERIES": "_mask_keep_edges_series",
        "PASSPORT_NUMBER": "_mask_keep_edges_number",
        "DRIVER_LICENSE": "_mask_keep_edges",
        "INN": "_mask_keep_edges",
        "EMAIL": "_mask_email",
        "PHONE": "_mask_phone",
        "BANK_CARD": "_mask_card",
        "DATE_OF_BIRTH": "_mask_date",
        "PASSPORT_ISSUE_DATE": "_mask_date",
        "PASSPORT_DEPT_CODE": "_mask_dept_code",
        "CARD_CVV": "_mask_cvv",
        "CARD_PIN": "_mask_pin",
    }

    def _mask_keep_edges_series(self, original: str) -> str:
        """Серия паспорта → сохраняем первые 2 цифры."""
        return self._mask_keep_edges(original, keep_start=2, keep_end=0)

    def _mask_keep_edges_number(self, original: str) -> str:
        """Номер паспорта → сохраняем последние 2 цифры."""
        return self._mask_keep_edges(original, keep_start=0, keep_end=2)


# Глобальный экземпляр
masker = Masker()
