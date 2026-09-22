from dataclasses import dataclass, field
from typing import Any
import uuid


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

    def to_entity(self, token: str | None = None) -> "PIIEntity":
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


@dataclass(slots=True)
class RequestContext:
    """Request-scoped контекст: маппинг токенов -> оригиналы, счетчики и т.д."""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    token_to_original: dict[str, str] = field(default_factory=dict)
    original_to_token: dict[str, str] = field(default_factory=dict)
    entity_counters: dict[str, int] = field(default_factory=dict)
    entities: list[PIIMatch] = field(default_factory=list)

    def get_next_token(self, entity_type: str) -> str:
        """Генерирует следующий токен для типа сущности"""
        from app.config.settings import settings
        count = self.entity_counters.get(entity_type, 0) + 1
        self.entity_counters[entity_type] = count
        token = settings.masker_token_format.format(type=entity_type, index=count)
        return f"{settings.masker_token_prefix}{token}{settings.masker_token_suffix}"

    def register_mapping(self, token: str, original: str) -> None:
        """Регистрирует маппинг токен -> оригинал"""
        self.token_to_original[token] = original
        self.original_to_token[original] = token

    def get_original(self, token: str) -> str | None:
        return self.token_to_original.get(token)

    def get_token(self, original: str) -> str | None:
        return self.original_to_token.get(original)

    def clear(self) -> None:
        """Полная очистка контекста (для безопасности)"""
        self.token_to_original.clear()
        self.original_to_token.clear()
        self.entity_counters.clear()
        self.entities.clear()


# ContextVar для request-scoped контекста
from contextvars import ContextVar

request_context: ContextVar[RequestContext | None] = ContextVar("request_context", default=None)


def get_request_context() -> RequestContext:
    """Получить текущий request context или создать новый"""
    ctx = request_context.get()
    if ctx is None:
        ctx = RequestContext()
        request_context.set(ctx)
    return ctx


def set_request_context(ctx: RequestContext) -> None:
    request_context.set(ctx)


def clear_request_context() -> None:
    ctx = request_context.get()
    if ctx:
        ctx.clear()
    request_context.set(None)