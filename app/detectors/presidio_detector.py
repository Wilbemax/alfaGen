from __future__ import annotations
import logging
from typing import TYPE_CHECKING

from app.detectors.base import BaseDetector, DetectorConfig
from app.models.pii import PIIMatch

if TYPE_CHECKING:
    from presidio_analyzer import AnalyzerEngine, RecognizerResult

logger = logging.getLogger(__name__)


# Маппинг Presidio типов на наши типы ПДн
PRESIDIO_TO_PII_TYPE = {
    "PERSON": "PERSON",
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "CREDIT_CARD": "BANK_CARD",
    "DATE_TIME": "DATE_OF_BIRTH",
    "LOCATION": "ADDRESS",
    "NRP": "PASSPORT_NUMBER",
    "IBAN_CODE": "BANK_ACCOUNT",
    "IP_ADDRESS": "IP_ADDRESS",
    "URL": "URL",
}


class PresidioDetector(BaseDetector):
    """Детектор на основе Microsoft Presidio с русскими recognizers"""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        super().__init__(config)
        self._analyzer: "AnalyzerEngine | None" = None

    @property
    def name(self) -> str:
        return "presidio"

    @property
    def supported_entity_types(self) -> set[str]:
        return set(PRESIDIO_TO_PII_TYPE.values())

    async def initialize(self) -> None:
        """Инициализация Presidio AnalyzerEngine"""
        try:
            from presidio_analyzer import AnalyzerEngine, PatternRecognizer
            from presidio_analyzer.nlp_engine import NlpEngineProvider

            # Настройка NLP engine для русского языка
            nlp_configuration = {
                "nlp_engine_name": "spacy",
                "models": [
                    {"lang_code": "ru", "model_name": "ru_core_news_md"},
                    {"lang_code": "en", "model_name": "en_core_web_sm"},
                ],
            }
            provider = NlpEngineProvider(nlp_configuration=nlp_configuration)
            nlp_engine = provider.create_engine()

            self._analyzer = AnalyzerEngine(
                nlp_engine=nlp_engine,
                supported_languages=[self.config.language, "en"],
            )

            # Добавляем русские recognizers
            self._add_russian_recognizers()

            logger.info("Presidio detector initialized")
        except ImportError:
            logger.warning("Presidio not installed, detector disabled")
            self.config.enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize Presidio: {e}")
            self.config.enabled = False

    def _add_russian_recognizers(self) -> None:
        """Добавляет русские recognizers для специфичных типов"""
        if not self._analyzer:
            return

        from presidio_analyzer import PatternRecognizer, Pattern

        # ИНН (10 или 12 цифр)
        inn_pattern = Pattern(
            name="INN_RU",
            regex=r"\b\d{10}\b|\b\d{12}\b",
            score=0.85,
        )
        inn_recognizer = PatternRecognizer(
            supported_entity="INN",
            patterns=[inn_pattern],
            name="INN_RU_Recognizer",
        )
        self._analyzer.registry.add_recognizer(inn_recognizer)

        # Паспорт РФ (серия 4 цифры + номер 6 цифр)
        passport_pattern = Pattern(
            name="PASSPORT_RU",
            regex=r"\b\d{4}\s?\d{6}\b",
            score=0.9,
        )
        passport_recognizer = PatternRecognizer(
            supported_entity="PASSPORT_NUMBER",
            patterns=[passport_pattern],
            name="PASSPORT_RU_Recognizer",
        )
        self._analyzer.registry.add_recognizer(passport_recognizer)

        # Код подразделения (XXX-XXX)
        dept_code_pattern = Pattern(
            name="DEPT_CODE_RU",
            regex=r"\b\d{3}-\d{3}\b",
            score=0.9,
        )
        dept_recognizer = PatternRecognizer(
            supported_entity="PASSPORT_DEPT_CODE",
            patterns=[dept_code_pattern],
            name="DEPT_CODE_RU_Recognizer",
        )
        self._analyzer.registry.add_recognizer(dept_recognizer)

        # Водительское удостоверение (серия 4 цифры + номер 6 цифр)
        driver_license_pattern = Pattern(
            name="DRIVER_LICENSE_RU",
            regex=r"\b\d{4}\s?\d{6}\b",
            score=0.8,
        )
        driver_recognizer = PatternRecognizer(
            supported_entity="DRIVER_LICENSE",
            patterns=[driver_license_pattern],
            name="DRIVER_LICENSE_RU_Recognizer",
        )
        self._analyzer.registry.add_recognizer(driver_recognizer)

        # CVV (3 цифры)
        cvv_pattern = Pattern(
            name="CVV_RU",
            regex=r"\b\d{3}\b",
            score=0.6,
        )
        cvv_recognizer = PatternRecognizer(
            supported_entity="CARD_CVV",
            patterns=[cvv_pattern],
            name="CVV_RU_Recognizer",
        )
        self._analyzer.registry.add_recognizer(cvv_recognizer)

    async def detect(self, text: str) -> list[PIIMatch]:
        if not self.config.enabled or not self._analyzer:
            return []

        try:
            # Presidio AnalyzerEngine - синхронный, запускаем в thread pool
            import asyncio
            results: list[RecognizerResult] = await asyncio.to_thread(
                self._analyzer.analyze,
                text=text,
                language=self.config.language,
            )

            matches: list[PIIMatch] = []
            for result in results:
                pii_type = PRESIDIO_TO_PII_TYPE.get(result.entity_type)
                if pii_type is None:
                    continue

                confidence = result.score
                if confidence < self.config.confidence_threshold:
                    continue

                matches.append(PIIMatch(
                    entity_type=pii_type,
                    text=text[result.start:result.end],
                    start=result.start,
                    end=result.end,
                    confidence=confidence,
                    detector_name=self.name,
                    metadata={"presidio_type": result.entity_type},
                ))

            return matches

        except Exception as e:
            logger.error(f"Presidio detection error: {e}")
            return []