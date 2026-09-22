from __future__ import annotations

from dataclasses import dataclass, field

from app.models.pii import PIIMatch


@dataclass(slots=True)
class MaskResult:
    """Результат маскирования с позициями, необходимыми для демаскирования."""

    masked_text: str
    spans: list[tuple[int, int, str]] = field(default_factory=list)
    entity_types: list[str] = field(default_factory=list)


class Masker:
    """Маскирует принятые полуинтервалы, не раскрывая символы внутри них."""

    def mask(self, text: str, entities: list[PIIMatch]) -> MaskResult:
        if not entities:
            return MaskResult(masked_text=text)

        selected = self._validated_non_overlapping(text, entities)
        if not selected:
            return MaskResult(masked_text=text)

        characters = list(text)
        spans: list[tuple[int, int, str]] = []
        entity_types: list[str] = []
        for entity in selected:
            original = text[entity.start:entity.end]
            characters[entity.start:entity.end] = "*" * (entity.end - entity.start)
            spans.append((entity.start, entity.end, original))
            entity_types.append(entity.entity_type)

        return MaskResult("".join(characters), spans, entity_types)

    def unmask(self, masked_text: str, spans: list[tuple[int, int, str]]) -> str:
        if not spans:
            return masked_text
        characters = list(masked_text)
        for start, end, original in spans:
            if 0 <= start <= end <= len(characters) and end - start == len(original):
                characters[start:end] = original
        return "".join(characters)

    @staticmethod
    def _validated_non_overlapping(text: str, entities: list[PIIMatch]) -> list[PIIMatch]:
        selected: list[PIIMatch] = []
        last_end = 0
        for entity in sorted(entities, key=lambda item: (item.start, item.end)):
            if not (0 <= entity.start < entity.end <= len(text)):
                continue
            if entity.start < last_end:
                continue
            if text[entity.start:entity.end] != entity.text:
                continue
            selected.append(entity)
            last_end = entity.end
        return selected


masker = Masker()
