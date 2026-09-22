from __future__ import annotations
import logging
from typing import TYPE_CHECKING

from app.detectors.base import BaseDetector, DetectorConfig
from app.models.pii import PIIMatch

if TYPE_CHECKING:
    from natasha import (
        Segmenter,
        MorphVocab,
        NewsEmbedding,
        NewsNERTagger,
        Doc,
        Span,
    )

logger = logging.getLogger(__name__)


# Маппинг Natasha тегов на наши типы ПДн
NATASHA_TO_PII_TYPE = {
    "PER": "PERSON",
    "LOC": "PLACE_OF_BIRTH",  # Может быть местом рождения
    "ORG": "ORGANIZATION",  # Не ПДн, но может быть полезно
}


class NatashaDetector(BaseDetector):
    """Детектор на основе Natasha (русский NER)"""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        super().__init__(config)
        self._segmenter = None
        self._morph_vocab = None
        self._emb = None
        self._ner_tagger = None

    @property
    def name(self) -> str:
        return "natasha"

    @property
    def supported_entity_types(self) -> set[str]:
        return {"PERSON", "PLACE_OF_BIRTH", "ORGANIZATION"}

    async def initialize(self) -> None:
        """Ленивая инициализация тяжелых моделей Natasha"""
        try:
            from natasha import (
                Segmenter,
                MorphVocab,
                NewsEmbedding,
                NewsNERTagger,
            )
            self._segmenter = Segmenter()
            self._morph_vocab = MorphVocab()
            self._emb = NewsEmbedding()
            self._ner_tagger = NewsNERTagger(self._emb)
            logger.info("Natasha detector initialized")
        except ImportError:
            logger.warning("Natasha not installed, detector disabled")
            self.config.enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize Natasha: {e}")
            self.config.enabled = False

    async def detect(self, text: str) -> list[PIIMatch]:
        if not self.config.enabled or not self._ner_tagger:
            return []

        try:
            from natasha import Doc

            doc = Doc(text)
            doc.segment(self._segmenter)
            doc.tag_ner(self._ner_tagger)

            matches: list[PIIMatch] = []
            for span in doc.spans:
                if span.type not in NATASHA_TO_PII_TYPE:
                    continue

                pii_type = NATASHA_TO_PII_TYPE[span.type]
                # Natasha не дает confidence, используем эвристику
                confidence = self._estimate_confidence(span, text)

                if confidence >= self.config.confidence_threshold:
                    matches.append(PIIMatch(
                        entity_type=pii_type,
                        text=span.text,
                        start=span.start,
                        end=span.stop,
                        confidence=confidence,
                        detector_name=self.name,
                        metadata={"natasha_type": span.type},
                    ))

            return matches

        except Exception as e:
            logger.error(f"Natasha detection error: {e}")
            return []

    def _estimate_confidence(self, span: "Span", text: str) -> float:
        """Эвристическая оценка уверенности для Natasha"""
        # Базовая уверенность
        confidence = 0.85

        # Повышаем для имен с отчеством (Иван Иванович)
        words = span.text.split()
        if len(words) >= 3 and any(w.endswith(("вич", "вна", "ич", "на")) for w in words[1:]):
            confidence = 0.95

        # Повышаем для длинных имен
        if len(span.text) > 15:
            confidence = min(confidence + 0.05, 0.98)

        # Понижаем для однобуквенных/очень коротких
        if len(span.text) < 3:
            confidence = 0.6

        return confidence