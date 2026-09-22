from __future__ import annotations

import logging
import time

from app.core.masker import masker
from app.core.payload_store import PayloadRecord, payload_store
from app.detectors.base import DetectorConfig
from app.detectors.regex_detector import RegexDetector

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
                logger.info(
                    "process_mask_retry",
                    extra={"payload_id": payload_id, "duration_ms": round((time.monotonic() - start_time) * 1000, 2)},
                )
                return existing.masked_text
            if payload == existing.masked_text:
                # Демаскирование: возвращаем исходник
                logger.info(
                    "process_unmask",
                    extra={"payload_id": payload_id, "duration_ms": round((time.monotonic() - start_time) * 1000, 2)},
                )
                return existing.original_text

        # Новый payload_id или payload не совпал ни с исходником, ни с маской
        result = await self._mask(payload, payload_id)
        logger.info(
            "process_mask",
            extra={"payload_id": payload_id, "duration_ms": round((time.monotonic() - start_time) * 1000, 2)},
        )
        return result

    async def _mask(self, payload: str, payload_id: str) -> str:
        """Маскирование: детекция + маска + сохранение соответствия."""
        entities = await self.detector(payload)
        mask_result = masker.mask(payload, entities)

        await payload_store.put(
            payload_id,
            PayloadRecord(
                original_text=payload,
                masked_text=mask_result.masked_text,
                entity_types=mask_result.entity_types,
            ),
        )
        return mask_result.masked_text


# Глобальный экземпляр pipeline
pipeline = Pipeline()