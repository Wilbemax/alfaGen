from __future__ import annotations

import re

# Разделители токенов: пробелы и знаки препинания
_TOKEN_SPLIT = re.compile(r"[\s,.;:!?()\[\]{}\"'«»—–-]+")


def count_tokens(text: str) -> int:
    """Подсчитывает количество токенов в тексте.

    Токен — слово или отдельный знак препинания. Для русского текста это
    близко к оценке LLM-токенизации. Используется для лимита длины запроса.
    """
    if not text:
        return 0
    parts = [p for p in _TOKEN_SPLIT.split(text) if p]
    return len(parts)
