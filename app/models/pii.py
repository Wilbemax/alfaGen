from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class PIIMatch:
    """Внутреннее представление найденной сущности ПДн (до превращения в PIIEntity)"""
    entity_type: str
    text: str
    start: int
    end: int
    confidence: float
    detector_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_entity(self, token: str | None = None) -> "PIIEntity":  # noqa: F821
        from app.models.request import PIIEntity, PIIEntityType
        # Если токен не передан явно, берем из metadata (если есть)
        resolved_token = token or self.metadata.get("token")
        return PIIEntity(
            type=PIIEntityType(self.entity_type),
            text=self.text,
            start=self.start,
            end=self.end,
            confidence=self.confidence,
            detector=self.detector_name,
            token=resolved_token,
        )
