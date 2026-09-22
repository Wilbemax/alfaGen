from __future__ import annotations

import hashlib
import logging
import time
from collections import Counter
from typing import Any

from app.config.settings import pii_rules, settings
from app.core.masker import MaskResult, masker
from app.core.payload_store import (
    PayloadRecord,
    PayloadStore,
    PayloadStoreError,
    normalize_scope,
    payload_store,
)
from app.detectors.cascade import CascadeDetector
from app.utils.metrics import observe_entities, observe_tokens
from app.utils.tokenizer import count_tokens

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Ошибка в pipeline обработки"""


class Pipeline:
    """
    Pipeline по контракту AlfaSonar.

    Направление определяется по payload_id и содержимому payload:
      - новый payload_id — маскирование (payload = исходная строка), возвращает маску
        и сохраняет пару «исходник ↔ маска» по payload_id;
      - тот же payload_id и payload == сохранённый исходник — ретрай маскирования,
        возвращает ту же маску без пересчёта;
      - тот же payload_id и payload == сохранённая маска — демаскирование,
        возвращает исходную строку (если demask_enabled);
      - чужой текст с известным payload_id — возвращает сохранённую маску,
        не перезаписывая пару.
    """

    def __init__(self, store: PayloadStore | None = None) -> None:
        self.cascade = CascadeDetector()
        self._initialized = False
        self._store = store if store is not None else payload_store

    async def initialize(self) -> None:
        """Инициализация каскада детекторов и хранилища."""
        if self._initialized:
            return
        await self.cascade.initialize()
        await self._store.initialize()
        self._initialized = True
        logger.info("Pipeline initialized")

    async def close(self) -> None:
        """Закрытие ресурсов хранилища."""
        await self._store.close()

    def _get_profile(self, system_id: str | None) -> dict[str, Any]:
        """Профиль системы из pii_rules."""
        return pii_rules.get_system_profile(system_id)

    async def process(
        self,
        payload: str,
        payload_id: str,
        system_id: str | None = None,
    ) -> str:
        """
        Обрабатывает payload по payload_id.
        Возвращает маску (новый id / ретрай) или исходную строку (демаскирование).
        """
        start_time = time.monotonic()
        profile = self._get_profile(system_id)

        if not self._initialized:
            await self.initialize()

        scope = normalize_scope(system_id)
        try:
            existing = await self._store.get(scope, payload_id)
        except PayloadStoreError:
            raise PipelineError("storage_error") from None
        if existing is not None:
            if payload == existing.original_text:
                # Ретрай маскирования: возвращаем ту же маску, не пересчитываем
                self._log_entities("process_mask_retry", existing.entity_types, start_time, payload_id)
                return existing.masked_text
            if payload == existing.masked_text:
                if profile.get("demask_enabled", False):
                    # Демаскирование: возвращаем исходник
                    self._log_entities("process_unmask", existing.entity_types, start_time, payload_id)
                    return existing.original_text
                # Демаскирование выключено: возвращаем маску
                self._log_entities("process_mask_retry", existing.entity_types, start_time, payload_id)
                return existing.masked_text
            # Чужой текст с известным payload_id: возвращаем сохранённую маску
            self._log_entities("process_mask_retry", existing.entity_types, start_time, payload_id)
            return existing.masked_text

        # Новый payload_id: маскирование
        result = await self._mask(payload, payload_id, profile, scope)
        self._log_entities("process_mask", result.entity_types, start_time, payload_id)
        return result.masked_text

    def _log_entities(self, event: str, entity_types: list[str], start_time: float, payload_id: str) -> None:
        """Логирует типы найденных ПДн и их число. payload/result в лог не попадают."""
        counts = Counter(entity_types)
        logger.info(
            event,
            extra={
                "payload_id": hashlib.sha256(payload_id.encode("utf-8")).hexdigest()[:8],
                "duration_ms": round((time.monotonic() - start_time) * 1000, 2),
                "entity_count": len(entity_types),
                "entity_types": dict(counts),
            },
        )

    async def _mask(
        self,
        payload: str,
        payload_id: str,
        profile: dict[str, Any],
        scope: str,
    ) -> MaskResult:
        """Маскирование: детекция + маска + сохранение соответствия."""
        allowed_types = profile.get("enabled_entity_types")
        if isinstance(allowed_types, list):
            allowed_types = set(allowed_types)

        try:
            entities = await self.cascade.detect(payload, allowed_types=allowed_types)
        except Exception:
            raise PipelineError("detection failed") from None

        if len(entities) > settings.pipeline_max_entities_per_request:
            raise PipelineError("entity_limit_exceeded")

        mask_result = masker.mask(payload, entities)

        # Метрики: счётчик сущностей по типам и обработанные токены
        observe_entities(mask_result.entity_types)
        observe_tokens(count_tokens(payload))

        try:
            await self._store.put(
                scope,
                payload_id,
                PayloadRecord(
                    original_text=payload,
                    masked_text=mask_result.masked_text,
                    entity_types=mask_result.entity_types,
                ),
            )
        except PayloadStoreError:
            raise PipelineError("storage_error") from None
        return mask_result


# Глобальный экземпляр pipeline
pipeline = Pipeline()
