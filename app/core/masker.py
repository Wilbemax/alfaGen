from __future__ import annotations
import re
from dataclasses import dataclass

from app.models.pii import RequestContext, PIIMatch


@dataclass(slots=True)
class MaskResult:
    """Результат маскирования"""
    masked_text: str
    entities: list[PIIMatch]


class Masker:
    """
    Маскирование и демаскирование текста.
    Использует токены вида [TYPE_INDEX] для замены ПДн.
    Токены генерируются через request-scoped контекст, поэтому уникальны в рамках запроса.
    """

    def mask(self, text: str, entities: list[PIIMatch], context: RequestContext) -> MaskResult:
        """
        Маскирует сущности в тексте.
        Сортирует сущности по позиции (справа налево) чтобы не сбить индексы.
        """
        if not entities:
            return MaskResult(masked_text=text, entities=[])

        # Сортируем по start позиции DESC (справа налево)
        sorted_entities = sorted(entities, key=lambda e: e.start, reverse=True)

        masked_text = text
        masked_entities: list[PIIMatch] = []

        for entity in sorted_entities:
            # Генерируем токен через request-scoped контекст
            token = context.get_token(entity.text)
            if token is None:
                token = context.get_next_token(entity.entity_type)
                context.register_mapping(token, entity.text)

            # Заменяем в тексте
            masked_text = (
                masked_text[:entity.start] + token + masked_text[entity.end:]
            )

            # Создаем новую сущность с токеном
            masked_entity = PIIMatch(
                entity_type=entity.entity_type,
                text=entity.text,
                start=entity.start,
                end=entity.start + len(token),
                confidence=entity.confidence,
                detector_name=entity.detector_name,
                metadata={**entity.metadata, "token": token},
            )
            masked_entities.append(masked_entity)

        # Возвращаем сущности в исходном порядке (слева направо)
        masked_entities.reverse()
        return MaskResult(masked_text=masked_text, entities=masked_entities)

    def unmask(self, text: str, context: RequestContext) -> str:
        """
        Демаскирует текст, заменяя токены на оригиналы.
        Использует контекст запроса для поиска оригиналов.
        """
        if not context.token_to_original:
            return text

        # Сортируем токены по длине DESC (чтобы длинные токены не ломали короткие)
        sorted_tokens = sorted(
            context.token_to_original.keys(),
            key=len,
            reverse=True
        )

        result = text
        for token in sorted_tokens:
            original = context.token_to_original[token]
            # Экранируем спецсимволы для regex
            escaped_token = re.escape(token)
            result = re.sub(escaped_token, original, result)

        return result

    def mask_for_llm(self, text: str, entities: list[PIIMatch], context: RequestContext) -> str:
        """Удобный метод: маскирует и возвращает только текст для LLM"""
        return self.mask(text, entities, context).masked_text

    def unmask_from_llm(self, text: str, context: RequestContext) -> str:
        """Удобный метод: демаскирует ответ LLM"""
        return self.unmask(text, context)


# Глобальный экземпляр
masker = Masker()