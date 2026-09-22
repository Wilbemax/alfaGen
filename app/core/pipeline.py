from __future__ import annotations

import logging
import time
from collections import Counter

from app.core.masker import MaskResult, masker
from app.core.payload_store import PayloadRecord, payload_store
from app.detectors.base import DetectorConfig
from app.detectors.regex_detector import RegexDetector
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
        возвращает исходную строку;
      - если ПДн нет и маска совпала с исходником, оба шага возвращают эту же строку.

    Горячий путь использует только regex-детектор и не обращается к LLM,
    не загружает Natasha или Presidio.
    """

    def __init__(self) -> None:
        self.detector = RegexDetector(DetectorConfig(enabled=True, confidence_threshold=0.5))
        self._initialized = False

    async def initialize(self) -> None:
        """Инициализация regex-детектора и хранилища."""
        if self._initialized:
            return
        await self.detector.initialize()
        await payload_store.initialize()
        self._initialized = True
        logger.info("Pipeline initialized (regex-only hot path)")

    async def close(self) -> None:
        """Закрытие ресурсов хранилища."""
        await payload_store.close()

    async def process(self, payload: str, payload_id: str) -> str:
        """
        Обрабатывает payload по payload_id.
        Возвращает маску (новый id / ретрай) или исходную строку (демаскирование).
        """
        start_time = time.monotonic()

        if not self._initialized:
            await self.initialize()

        existing = await payload_store.get(payload_id)
        if existing is not None:
            if payload == existing.original_text:
                # Ретрай маскирования: возвращаем ту же маску, не пересчитываем
                self._log_entities("process_mask_retry", existing.entity_types, start_time, payload_id)
                return existing.masked_text
            if payload == existing.masked_text:
                # Демаскирование: возвращаем исходник
                self._log_entities("process_unmask", existing.entity_types, start_time, payload_id)
                return existing.original_text

        # Новый payload_id или payload не совпал ни с исходником, ни с маской
        result = await self._mask(payload, payload_id)
        self._log_entities("process_mask", result.entity_types, start_time, payload_id)
        return result.masked_text

    def _log_entities(self, event: str, entity_types: list[str], start_time: float, payload_id: str) -> None:
        """Логирует типы найденных ПДн и их число. payload/result в лог не попадают."""
        counts = Counter(entity_types)
        logger.info(
            event,
            extra={
                "payload_id": payload_id,
                "duration_ms": round((time.monotonic() - start_time) * 1000, 2),
                "entity_count": len(entity_types),
                "entity_types": dict(counts),
            },
        )

    async def _mask(self, payload: str, payload_id: str) -> MaskResult:
        """Маскирование: детекция + маска + сохранение соответствия."""
        entities = await self.detector(payload)
        mask_result = masker.mask(payload, entities)

        # Метрики: счётчик сущностей по типам и обработанные токены
        observe_entities(mask_result.entity_types)
        observe_tokens(count_tokens(payload))

        await payload_store.put(
            payload_id,
            PayloadRecord(
                original_text=payload,
                masked_text=mask_result.masked_text,
                entity_types=mask_result.entity_types,
            ),
        )
        return mask_result


# Глобальный экземпляр pipeline
pipeline = Pipeline()
