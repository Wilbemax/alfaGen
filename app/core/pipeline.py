from __future__ import annotations
import logging
import time
from typing import TYPE_CHECKING

from app.config.settings import settings, pii_rules
from app.core.masker import masker
from app.detectors.composite import CompositeDetector
from app.models.pii import (
    PIIMatch,
    RequestContext,
    get_request_context,
    set_request_context,
    clear_request_context,
)
from app.models.request import (
    ProcessRequest,
    ProcessResponse,
    DetectOnlyResponse,
    MaskOnlyResponse,
    PIIEntity,
    ProcessingMode,
)
from app.services.llm_client import llm_client, LLMClientError

if TYPE_CHECKING:
    from app.config.settings import settings as app_settings

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Ошибка в pipeline обработки"""


class Pipeline:
    """
    Главный pipeline обработки:
    detect -> mask -> LLM -> unmask
    """

    def __init__(self) -> None:
        self.detector = CompositeDetector()
        self._initialized = False

    async def initialize(self) -> None:
        """Инициализация всех компонентов"""
        if self._initialized:
            return

        self.detector.configure(
            strategy=settings.composite_strategy,
            deduplicate=settings.composite_deduplicate,
            min_confidence=settings.composite_min_confidence,
            natasha_enabled=settings.natasha_enabled,
            presidio_enabled=settings.presidio_enabled,
            regex_enabled=settings.regex_enabled,
        )
        await self.detector.initialize()
        await llm_client.initialize()
        self._initialized = True
        logger.info("Pipeline initialized")

    async def process(self, request: ProcessRequest) -> ProcessResponse | DetectOnlyResponse | MaskOnlyResponse:
        """
        Обрабатывает запрос согласно режиму.
        Создает request-scoped контекст, который очищается после обработки.
        """
        start_time = time.monotonic()

        # Ленивая инициализация (на случай, если lifespan не был вызван)
        if not self._initialized:
            await self.initialize()

        # Создаем request-scoped контекст
        ctx = RequestContext()
        set_request_context(ctx)

        try:
            # Получаем конфигурацию системы
            system_config = pii_rules.get_system_config(request.system_id)
            enabled_types = set(system_config.get("enabled_entity_types", []))
            exclusions = system_config.get("exclusions", {})

            # Проверяем длину текста
            if len(request.text) > settings.pipeline_max_text_length:
                raise PipelineError(f"Text too long: {len(request.text)} > {settings.pipeline_max_text_length}")

            # Режим detect_only
            if request.mode == ProcessingMode.DETECT_ONLY:
                entities = await self._detect(request.text, enabled_types, exclusions)
                return DetectOnlyResponse(
                    entities=[e.to_entity() for e in entities],
                    processing_time_ms=(time.monotonic() - start_time) * 1000,
                )

            # Режим mask_only
            if request.mode == ProcessingMode.MASK_ONLY:
                entities = await self._detect(request.text, enabled_types, exclusions)
                mask_result = masker.mask(request.text, entities, ctx)
                return MaskOnlyResponse(
                    masked_text=mask_result.masked_text,
                    entities=[e.to_entity() for e in mask_result.entities],
                    processing_time_ms=(time.monotonic() - start_time) * 1000,
                )

            # Режим unmask_only
            if request.mode == ProcessingMode.UNMASK_ONLY:
                # В этом режиме контекст должен быть передан через metadata
                # (для простоты считаем, что текст уже замаскирован и контекст пуст)
                unmasked = masker.unmask(request.text, ctx)
                return ProcessResponse(
                    original_text=request.text,
                    masked_text=request.text,
                    llm_response="",
                    unmasked_response=unmasked,
                    entities=[],
                    processing_time_ms=(time.monotonic() - start_time) * 1000,
                    llm_time_ms=0,
                )

            # Полный режим: detect -> mask -> LLM -> unmask
            return await self._process_full(request, ctx, enabled_types, exclusions, start_time)

        finally:
            # Всегда очищаем контекст (даже при ошибке)
            clear_request_context()

    async def _detect(
        self,
        text: str,
        enabled_types: set[str],
        exclusions: dict,
    ) -> list[PIIMatch]:
        """Детекция ПДн с фильтрацией по типам и исключениям"""
        matches = await self.detector(text)
        matches = self.detector.filter_by_types(matches, enabled_types)
        # Разрешаем перекрытия ПОСЛЕ фильтрации по типам,
        # чтобы не терять менее специфичные типы (например, PASSPORT_SERIES)
        matches = self.detector.resolve_overlaps(matches)
        matches = self.detector.apply_exclusions(matches, exclusions)

        # Ограничиваем количество сущностей
        if len(matches) > settings.pipeline_max_entities_per_request:
            matches = matches[: settings.pipeline_max_entities_per_request]

        return matches

    async def _process_full(
        self,
        request: ProcessRequest,
        ctx: RequestContext,
        enabled_types: set[str],
        exclusions: dict,
        start_time: float,
    ) -> ProcessResponse:
        """Полный pipeline: detect -> mask -> LLM -> unmask"""
        # 1. Детекция
        entities = await self._detect(request.text, enabled_types, exclusions)

        # 2. Маскирование
        mask_result = masker.mask(request.text, entities, ctx)
        masked_text = mask_result.masked_text

        # 3. Отправка в LLM
        llm_start = time.monotonic()
        try:
            llm_response = await llm_client.generate(
                prompt=masked_text,
                system_prompt=request.llm_prompt,
            )
        except LLMClientError as e:
            logger.error(f"LLM error: {e}")
            raise PipelineError(f"LLM processing failed: {e}") from e
        llm_time_ms = (time.monotonic() - llm_start) * 1000

        # 4. Демаскирование ответа LLM
        unmasked_response = masker.unmask(llm_response, ctx)

        return ProcessResponse(
            original_text=request.text,
            masked_text=masked_text,
            llm_response=llm_response,
            unmasked_response=unmasked_response,
            entities=[e.to_entity() for e in mask_result.entities],
            processing_time_ms=(time.monotonic() - start_time) * 1000,
            llm_time_ms=llm_time_ms,
        )


# Глобальный экземпляр pipeline
pipeline = Pipeline()