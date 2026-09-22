from __future__ import annotations

import logging
import re
from contextlib import suppress
from typing import Any

# Паттерны для обнаружения ПДн в логах
PII_PATTERNS: list[re.Pattern] = [
    # Email
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    # Телефон
    re.compile(r"(?:\+7|8|7)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}"),
    # Банковская карта
    re.compile(r"(?:\d{4}[\s\-]?){3}\d{4}"),
    # ИНН
    re.compile(r"(?<!\d)\d{12}(?!\d)"),
    # Паспорт (серия + номер)
    re.compile(r"\d{4}\s?\d{6}"),
    # Код подразделения
    re.compile(r"\d{3}-\d{3}"),
    # CVV
    re.compile(r"(?i)(?:cvv|cvc)[\s:]*\d{3}\b"),
    # Пин-код
    re.compile(r"(?i)(?:пин[\s-]?код|pin)[\s:]*\d{4}\b"),
    # Дата рождения
    re.compile(r"(?<!\d)(?:0[1-9]|[12]\d|3[01])\.(?:0[1-9]|1[0-2])\.(?:19|20)\d{2}"),
    # ФИО (2-3 слова с заглавной буквы)
    re.compile(r"(?<![А-ЯЁа-яё])(?:[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){1,2})(?![а-яё])"),
]


class PIISanitizingFilter(logging.Filter):
    """
    Logging filter, который заменяет ПДн на [REDACTED] в лог-сообщениях.
    Гарантирует, что исходные ПДн никогда не попадут в логи.
    """

    def __init__(self, pii_patterns: str | None = None) -> None:
        super().__init__()
        self._patterns = PII_PATTERNS
        if pii_patterns:
            with suppress(re.error):
                self._patterns = [re.compile(p) for p in pii_patterns.split(",")]

    def filter(self, record: logging.LogRecord) -> bool:
        """Санитизирует сообщение лога"""
        if hasattr(record, "msg") and isinstance(record.msg, str):
            record.msg = self._sanitize(record.msg)

        # Санитизируем аргументы
        if record.args:
            record.args = self._sanitize_args(record.args)

        # Санитизируем дополнительные поля
        for key in ("text", "message", "prompt", "content", "body", "input", "output"):
            if hasattr(record, key):
                value = getattr(record, key)
                if isinstance(value, str):
                    setattr(record, key, self._sanitize(value))

        return True

    def _sanitize(self, text: str) -> str:
        """Заменяет ПДн на [REDACTED]"""
        result = text
        for pattern in self._patterns:
            result = pattern.sub("[REDACTED]", result)
        return result

    def _sanitize_args(self, args: Any) -> Any:
        """Санитизирует аргументы лога"""
        if isinstance(args, tuple):
            return tuple(self._sanitize_arg(a) for a in args)
        if isinstance(args, dict):
            return {k: self._sanitize_arg(v) for k, v in args.items()}
        return args

    def _sanitize_arg(self, arg: Any) -> Any:
        if isinstance(arg, str):
            return self._sanitize(arg)
        if isinstance(arg, (dict, list, tuple)):
            return self._sanitize_args(arg)
        return arg


def sanitize_text(text: str) -> str:
    """Утилита для санитизации текста вне logging"""
    result = text
    for pattern in PII_PATTERNS:
        result = pattern.sub("[REDACTED]", result)
    return result


def sanitize_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Санитизирует все строковые значения в словаре"""
    result: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, str):
            result[key] = sanitize_text(value)
        elif isinstance(value, dict):
            result[key] = sanitize_dict(value)
        elif isinstance(value, list):
            result[key] = [sanitize_text(v) if isinstance(v, str) else v for v in value]
        else:
            result[key] = value
    return result
