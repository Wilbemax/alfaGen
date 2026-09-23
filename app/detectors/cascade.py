from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from app.config.settings import settings
from app.detectors.base import DetectorConfig
from app.detectors.context_filter import ContextFilter
from app.detectors.regex_detector import RegexDetector

if TYPE_CHECKING:
    from app.models.pii import PIIMatch

logger = logging.getLogger(__name__)


class CascadeDetector:
    """Каскад regex → Natasha → context_filter с безопасной деградацией."""

    def __init__(
        self,
        regex_detector: RegexDetector | None = None,
        context_filter: ContextFilter | None = None,
        natasha_detector: object | None = None,
    ) -> None:
        self._regex = regex_detector or RegexDetector(
            DetectorConfig(enabled=True, confidence_threshold=0.5)
        )
        self._context_filter = context_filter or ContextFilter()
        self._natasha = natasha_detector
        self._initialized: bool = False
        self._natasha_available: bool = False

        if self._natasha is None and settings.natasha_enabled:
            try:
                from app.detectors.natasha_detector import NatashaDetector

                self._natasha = NatashaDetector()
            except Exception:
                logger.warning("NatashaDetector unavailable, cascade will use regex only")
                self._natasha = None

    @property
    def natasha_available(self) -> bool:
        return self._natasha_available

    async def initialize(self) -> None:
        """Инициализация regex-детектора и (если доступен) Natasha."""
        if self._initialized:
            return
        await self._regex.initialize()
        if self._natasha is not None:
            try:
                init = getattr(self._natasha, "initialize", None)
                if init is not None:
                    await init()
                self._natasha_available = True
            except Exception:
                logger.warning("Natasha initialization failed, cascade will use regex only")
                self._natasha_available = False
        self._initialized = True

    async def detect(
        self,
        text: str,
        allowed_types: set[str] | None = None,
    ) -> list[PIIMatch]:
        """Детекция через каскад с фильтрацией по типам."""
        start = time.monotonic()

        regex_matches = await self._regex.detect(text)
        occupied = [(m.start, m.end) for m in regex_matches]

        natasha_matches: list[PIIMatch] = []
        if self._natasha_available and self._natasha is not None:
            try:
                natasha_matches = await self._natasha.detect(
                    text,
                    occupied=occupied,
                    deadline_monotonic=start + settings.natasha_deadline_seconds,
                )
            except TypeError:
                try:
                    natasha_matches = await self._natasha.detect(text)
                except Exception:
                    logger.warning("Natasha detect failed, continuing with regex only")
                    natasha_matches = []
            except Exception:
                logger.warning("Natasha detect failed, continuing with regex only")
                natasha_matches = []

        combined = regex_matches + natasha_matches
        filtered = self._context_filter.filter(text, combined)

        if allowed_types is not None:
            filtered = [m for m in filtered if m.entity_type in allowed_types]

        return filtered
