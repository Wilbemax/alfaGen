from app.config.settings import settings


class TokenGenerator:
    """Генератор токенов маскирования вида [TYPE_INDEX]"""

    def __init__(self) -> None:
        self.prefix = settings.masker_token_prefix
        self.suffix = settings.masker_token_suffix
        self.format_template = settings.masker_token_format
        self._counters: dict[str, int] = {}

    def generate(self, entity_type: str) -> str:
        """Генерирует уникальный токен для типа сущности"""
        count = self._counters.get(entity_type, 0) + 1
        self._counters[entity_type] = count
        token_body = self.format_template.format(type=entity_type, index=count)
        return f"{self.prefix}{token_body}{self.suffix}"

    def reset(self) -> None:
        self._counters.clear()

    def get_counts(self) -> dict[str, int]:
        return self._counters.copy()


# Глобальный генератор (будет пересоздаваться на запрос через context)
token_generator = TokenGenerator()