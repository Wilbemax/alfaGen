from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.pii import PIIMatch


@dataclass(slots=True)
class DetectorConfig:
    """Конфигурация детектора"""
    enabled: bool = True
    confidence_threshold: float = 0.75
    language: str = "ru"
    priority: int = 100  # чем меньше, тем выше приоритет


class BaseDetector(ABC):
    """Базовый класс для всех детекторов ПДн"""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()
        self._initialized = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Уникальное имя детектора"""
        ...

    @property
    @abstractmethod
    def supported_entity_types(self) -> set[str]:
        """Множество поддерживаемых типов сущностей"""
        ...

    @abstractmethod
    async def initialize(self) -> None:
        """Асинхронная инициализация (загрузка моделей и т.д.)"""
        ...

    @abstractmethod
    async def detect(self, text: str) -> list["PIIMatch"]:
        """Детекция сущностей в тексте"""
        ...

    async def __call__(self, text: str) -> list["PIIMatch"]:
        if not self.config.enabled:
            return []
        if not self._initialized:
            await self.initialize()
            self._initialized = True
        return await self.detect(text)

    def filter_by_confidence(self, matches: list["PIIMatch"]) -> list["PIIMatch"]:
        """Фильтрация по порогу уверенности"""
        return [m for m in matches if m.confidence >= self.config.confidence_threshold]

    def filter_by_types(self, matches: list["PIIMatch"], allowed_types: set[str]) -> list["PIIMatch"]:
        """Фильтрация по разрешенным типам"""
        return [m for m in matches if m.entity_type in allowed_types]
